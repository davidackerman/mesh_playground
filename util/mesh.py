import os
import time
import warnings
import numpy as np
import pymeshlab
import trimesh
import navis
from navis import graph
import skeletor as sk
import dask
import networkx as nx
from xvfbwrapper import Xvfb

from util.precomputed_mesh import PrecomputedMeshReader


class MeshProcessor:
    def __init__(
        self,
        path: str,
        lod: int,
        min_branch_length: int = 500,
        close_holes=False,
        use_skeletons: bool = False,
        numberrays=128,
    ):
        self.reader = PrecomputedMeshReader(path, lod)
        self.path = self.reader.path
        self.is_file_path = self.reader.is_file_path
        self.session = self.reader.session
        self.lod = lod
        self.min_branch_length = min_branch_length
        self.use_skeletons = use_skeletons
        self.close_holes = close_holes
        self.numberrays = numberrays

    def get_mesh(self, id):
        # load and decode the mesh (handles sharded + unsharded layouts)
        all_vertices, all_faces = self.reader.load_vertices_faces(id)

        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(all_vertices, all_faces)
        ms.add_mesh(m)
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_repair_non_manifold_edges(
            method="Remove Faces"
        )  # sometimes this still leaves nonmanifold vertices
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=4)

        # iteratively close any boundary holes left after repair; stop once the
        # number of boundary edges stops decreasing (or hits zero)
        try:
            measures = ms.apply_filter("get_topological_measures")
            boundary_edges_prev = np.inf
            while 0 < measures["boundary_edges"] < boundary_edges_prev:
                boundary_edges_prev = measures["boundary_edges"]
                ms.meshing_close_holes(maxholesize=measures["boundary_edges"] + 1)
                measures = ms.apply_filter("get_topological_measures")
        except Exception as e:
            warnings.warn(f"hole-closing skipped for {id}: {e}")

        mesh = trimesh.Trimesh(
            ms.current_mesh().vertex_matrix(),
            ms.current_mesh().face_matrix(),
            process=False,
        )
        return mesh, ms

    @staticmethod
    def watertight_metrics(mesh):
        """Topology diagnostics for a (repaired) trimesh.

        A non-watertight mesh has unreliable ``volume`` (trimesh integrates over
        an open surface), so these fields let callers flag suspect meshes.
        """
        edges = mesh.edges_sorted
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return {
            "is_watertight": bool(mesh.is_watertight),
            "is_winding_consistent": bool(mesh.is_winding_consistent),
            "euler_number": int(mesh.euler_number),
            "num_boundary_edges": int((counts == 1).sum()),
            "num_vertices": int(len(mesh.vertices)),
            "num_faces": int(len(mesh.faces)),
        }

    def check_watertight(self, id):
        """Download mesh ``id`` and report watertightness diagnostics."""
        mesh, _ = self.get_mesh(id)
        return {"id": id, **self.watertight_metrics(mesh)}

    @dask.delayed
    def process_mesh(self, id):
        # os.system(f"touch {id}.txt")

        mesh, ms = self.get_mesh(id)
        # calculate general mesh properties
        metrics = {"id": id}
        metrics.update(self.watertight_metrics(mesh))
        metrics["volume"] = mesh.volume
        metrics["surface_area"] = mesh.area
        pic = mesh.principal_inertia_components
        pic_normalized = pic / np.sum(pic)
        _, ob = trimesh.bounds.oriented_bounds(mesh)
        ob_normalized = ob / np.sum(ob)
        for axis in range(3):
            metrics[f"pic_{axis}"] = pic[axis]
            metrics[f"pic_normalized_{axis}"] = pic_normalized[axis]
            metrics[f"ob_{axis}"] = ob[axis]
            metrics[f"ob_normalized_{axis}"] = ob_normalized[axis]
        # ms.calculat
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore")
        #     metrics["mean_curvature"] = np.nanmean(
        #         my_discrete_mean_curvature_measure(mesh)
        #     )

        # metrics["gaussian_curvature"] = np.nanmean(
        #     my_discrete_gaussian_curvature_measure(mesh)
        # )

        # mesh = mesh.process()
        vdisplay = Xvfb()
        vdisplay.start()

        try:
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_edges()
            for idx, metric in enumerate(["mean", "gaussian", "rms", "abs"]):
                ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=idx)
                vsa = ms.current_mesh().vertex_scalar_array()
                metrics[f"{metric}_curvature_mean"] = np.nanmean(vsa)
                metrics[f"{metric}_curvature_median"] = np.nanmedian(vsa)
                metrics[f"{metric}_curvature_std"] = np.nanstd(vsa)

            ms.compute_scalar_by_shape_diameter_function_per_vertex(
                numberrays=self.numberrays
            )
            vsa = ms.current_mesh().vertex_scalar_array()
            metrics["thickness_mean"] = np.nanmean(vsa)
            metrics["thickness_median"] = np.nanmedian(vsa)
            metrics["thickness_std"] = np.nanstd(vsa)

            # # center of each subdivided face offset inwards
            # points = mesh.triangles_center + (mesh.face_normals * -1e-4)
            # # use the original mesh for thickness as it is well constructed
            # metrics["thickness"] = np.nanmean(
            #     trimesh.proximity.thickness(mesh=mesh, points=points)
            # )
        except Exception as e:
            ms.save_current_mesh(f"{id}.ply")
            raise Exception(f"failed {id}") from e
        finally:
            vdisplay.stop()
        # for axis in range(3):
        #     metrics[f"axis_momenta_{axis}"] = measures["axis_momenta"][axis]
        #     metrics[f"axis_momenta_normalized_{axis}"] = axis_momenta_normalized[axis]
        # measures = ms.apply_filter("get_geometric_measures")
        # metrics["volume"] = measures["mesh_volume"]
        # metrics["surface_area"] = measures["surface_area"]
        # axis_momenta_normalized = measures["axis_momenta"] / np.sum(
        #     measures["axis_momenta"]
        # )
        # for axis in range(3):
        #     metrics[f"axis_momenta_{axis}"] = measures["axis_momenta"][axis]
        #     metrics[f"axis_momenta_normalized_{axis}"] = axis_momenta_normalized[axis]

        if self.use_skeletons:
            # calculate metrics using navis
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skeleton = sk.skeletonize.by_wavefront(
                    mesh,
                    waves=1,
                    step_size=2,
                    progress=False,
                )
                # skeleton = sk.skeletonize.by_tangent_ball(
                #     (ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix())
                # )
                sk.post.clean_up(skeleton, inplace=True)
                n = navis.TreeNeuron(skeleton, soma=None)
            navis.prune_twigs(
                n, size=self.min_branch_length, inplace=True, recursive=True
            )
            # metrics["branch_lengths"] = np.array(
            #     [graph.segment_length(n, s) for s in n.segments]
            # )
            num_branches = 0
            longest_path = 0
            fragments = []
            if n.n_nodes > 1:
                fragments = navis.split_into_fragments(
                    n, n=float("inf"), min_size=self.min_branch_length
                )
                # len(fragments)
                # print(fragments)
                # navis.plot3d(
                #     [
                #         n,
                #         fragments,
                #         trimesh.Trimesh(
                #             ms.current_mesh().vertex_matrix(),
                #             ms.current_mesh().face_matrix(),
                #         ),
                #     ]
                # )
                # print(n)
                longest_path = navis.longest_neurite(n, from_root=False).cable_length
                num_branches = n.n_branches
            metrics["num_fragments"] = len(fragments)
            metrics["num_branches"] = num_branches
            metrics["longest_path"] = longest_path
        # metrics["mesh"] = mesh
        # os.system(f"rm {id}.txt")
        # return measures["axis_momenta"]
        return metrics

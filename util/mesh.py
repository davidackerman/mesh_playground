import os
import time
import warnings
import requests
import struct
import numpy as np
import pymeshlab
import trimesh
import DracoPy
import navis
from navis import graph
import skeletor as sk
import dask
import networkx as nx
from xvfbwrapper import Xvfb


from requests.adapters import HTTPAdapter
from urllib3.response import HTTPResponse
class FileAdapter(HTTPAdapter):
    def send(self, request, *args, **kwargs):
        resp = HTTPResponse(body=open(request.url[7:], 'rb'), status=200, preload_content=False)
        return self.build_response(request, resp)

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
        self.is_file_path = False
        if not (path.startswith("http") or path.startswith("file")):
            self.is_file_path = True
            path = "file://" + path
        self.path = path
        self.lod = lod
        self.min_branch_length = min_branch_length
        self.use_skeletons = use_skeletons
        self.close_holes = close_holes
        self.numberrays = numberrays

        # for local data
        self.session = requests.Session()
        self.session.mount('file://', FileAdapter())

    
    def get_mesh(self, id):
        def get_lod_bytes(lod_byte_offset):
            # read mesh lod
            if self.is_file_path:
                # range requests don't work with file path
                with open(f"{self.path[7:]}/{id}", mode="rb") as f:
                    f.seek(lod_byte_offset[0])
                    lod_bytes = f.read(lod_byte_offset[-1]-lod_byte_offset[0])
            else:
                response = self.session.get(
                    f"{self.path}/{id}",
                    headers={"range": f"bytes={lod_byte_offset[0]}-{lod_byte_offset[-1]}"},
                )
                lod_bytes = response.content
            return lod_bytes
    
        def unpack_and_remove(datatype, num_elements, file_content):
            datatype = datatype * num_elements
            output = struct.unpack(datatype, file_content[0 : 4 * num_elements])
            file_content = file_content[4 * num_elements :]
            return np.array(output), file_content

        # get quantization and transform (for voxel size)
        response = self.session.get(f"{self.path}/info")
        meshes_info = response.json()
        vertex_quantization_bits = meshes_info["vertex_quantization_bits"]
        meshes_transform = meshes_info["transform"]
        meshes_transform += [0, 0, 0, 1]
        meshes_transform = np.reshape(meshes_transform, (4, 4))

        # get index file info
        response = self.session.get(f"{self.path}/{id}.index")
        index_file_content = response.content
        chunk_shape, index_file_content = unpack_and_remove("f", 3, index_file_content)
        grid_origin, index_file_content = unpack_and_remove("f", 3, index_file_content)
        num_lods, index_file_content = unpack_and_remove("I", 1, index_file_content)
        lod_scales, index_file_content = unpack_and_remove(
            "f", num_lods[0], index_file_content
        )
        vertex_offsets, index_file_content = unpack_and_remove(
            "f", num_lods[0] * 3, index_file_content
        )
        num_fragments_per_lod, index_file_content = unpack_and_remove(
            "I", num_lods[0], index_file_content
        )

        previous_lod_byte_offset = 0
        for current_lod in range(self.lod + 1):
            fragment_positions, index_file_content = unpack_and_remove(
                "I", num_fragments_per_lod[current_lod] * 3, index_file_content
            )
            fragment_positions = fragment_positions.reshape((3, -1)).T
            fragment_offsets, index_file_content = unpack_and_remove(
                "I", num_fragments_per_lod[current_lod], index_file_content
            )

            lod_byte_offset = (
                np.cumsum(np.array(fragment_offsets)) + previous_lod_byte_offset
            )
            lod_byte_offset = np.insert(lod_byte_offset, 0, previous_lod_byte_offset)
            previous_lod_byte_offset = lod_byte_offset[-1]  # end of previous lod
        mesh_fragments = []

        # read mesh lod
        lod_bytes = get_lod_bytes(lod_byte_offset)

        all_vertices = np.empty((0, 3))
        all_faces = np.empty((0, 3))

        for idx, fragment_offset in enumerate(fragment_offsets):
            if lod_byte_offset[idx] != lod_byte_offset[idx + 1]:  # nonempty chunk

                start = lod_byte_offset[idx] - lod_byte_offset[0]
                stop = lod_byte_offset[idx + 1] - lod_byte_offset[0] + 1
                drc_mesh = DracoPy.decode(lod_bytes[start:stop])  # response.content)
                vertices = drc_mesh.points
                faces = drc_mesh.faces

                vertices = (
                    grid_origin
                    + vertex_offsets[self.lod]
                    + chunk_shape
                    * (2**self.lod)
                    * (
                        fragment_positions[idx]
                        + vertices / (2**vertex_quantization_bits - 1)
                    )
                )
                # _ = trimesh.Trimesh(vertices, faces, process=False).export(
                #     f"{id}_{idx}.ply"
                # )
                all_faces = np.concatenate(
                    (all_faces, faces + all_vertices.shape[0]), axis=0
                )
                all_vertices = np.concatenate((all_vertices, vertices), axis=0)
        # mesh = trimesh.Trimesh(all_vertices, all_faces)
        # mesh.remove_duplicate_faces()
        # _ = mesh.export(f"{id}.ply")
        # while not mesh.is_watertight:
        #    mesh.fill_holes
        ms = pymeshlab.MeshSet()
        # # all_vertices -= np.mean(all_vertices, axis=0)

        m = pymeshlab.Mesh(all_vertices, all_faces)
        ms.add_mesh(m)
        ms.meshing_remove_duplicate_vertices()
        # # ms.meshing_merge_close_vertices(
        # #     threshold=pymeshlab.Percentage(
        # #         (5 / ms.current_mesh().bounding_box().diagonal()) * 10000,
        # #     )
        # # )  # meshing_remove_duplicate_vertices()  #
        ms.meshing_remove_duplicate_faces()
        # # mesh = trimesh.Trimesh(all_vertices, all_faces, process=False)
        # # _ = mesh.export(f"{id}.obj")
        # ms.save_current_mesh(f"{id}.ply")
        # measures = ms.apply_filter("get_topological_measures")
        # if measures["number_holes"] != 0:
        ms.meshing_repair_non_manifold_edges(
            method="Remove Faces"
            # method="Split Vertices", previously used split vertices
        )  # sometimes this still has nonmanifold vertices
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=4)
        #     ms.meshing_repair_non_manifold_vertices(vertdispratio=0.01)
        #     # not sure why this doesn't work on first try even though it does in ui

        try:
            measures = ms.apply_filter("get_topological_measures")
            boundary_edges_prev = np.Inf
            while 0 < measures["boundary_edges"] < boundary_edges_prev:
                boundary_edges_prev = measures["boundary_edges"]
                ms.meshing_close_holes(maxholesize=measures["boundary_edges"] + 1)
                measures = ms.apply_filter("get_topological_measures")
        except:
            pass
        # if self.close_holes:
        #     # we shouldnt have real "holes" in the newer approach, but "weird" holes due only to nonmanifold edges
        #     # in older approach for fab four we can have real holes, so we only want to fill holes when we have real holes
        #     # < 0 means something else going on with holes
        #     # HACK
        #     # then there are holes or something weird with mesh, try to fill holes.

        #     # seems to not always work on 1 iteration. also with the fab four datasets we have holes due to missing faces
        #     # but for newer ones, there shouldn't be any "real" holes like that i dont think?
        #     # ultimately we don't care if the mesh is watertight as long as we can get useful measures
        #     # our older meshes have holes
        #     measures = ms.apply_filter("get_topological_measures")
        #     while measures["number_holes"] != 0:
        #         ms.meshing_close_holes(maxholesize=measures["boundary_edges"] + 1)
        #         measures = ms.apply_filter("get_topological_measures")
        # for _ in range(2):
        # at this point our mesh shouldn't have any fillable holes, but may have nonmanifold edges, which pymeshlab will give errors for. but trimesh seems to produce results with
        # ms.save_current_mesh(f"{id}.ply")
        # ms.save_current_mesh(f"{id}.ply")
        # ms.meshing_isotropic_explicit_remeshing()  # to help with skeletonization later on
        mesh = trimesh.Trimesh(
            ms.current_mesh().vertex_matrix(),
            ms.current_mesh().face_matrix(),
            # process=False,
            process=False,
        )
        # broken_faces = trimesh.repair.broken_faces(mesh)
        # faces_to_remove = [True] * len(mesh.faces)
        # for broken_face in broken_faces:
        #    faces_to_remove[broken_face] = False
        # print(broken_faces)
        # mesh.update_faces(faces_to_remove)
        # _ = mesh.export(f"{id}.ply")
        # while not mesh.is_watertight:
        #    mesh.fill_holes()
        # measures = ms.apply_filter("get_topological_measures")
        return mesh, ms

    @dask.delayed
    def process_mesh(self, id):
        # os.system(f"touch {id}.txt")

        mesh, ms = self.get_mesh(id)
        # calculate gneral mesh properties
        metrics = {"id": id}
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
        except:
            ms.save_current_mesh(f"{id}.ply")
            raise Exception(f"failed {id}")
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

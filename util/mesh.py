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


class MeshProcessor:
    def __init__(
        self,
        path: str,
        lod: int,
        min_branch_length: int = 500,
        use_skeletons: bool = False,
    ):
        self.path = path
        self.lod = lod
        self.min_branch_length = min_branch_length
        self.use_skeletons = use_skeletons

    dask.delayed

    def get_mesh(self, id):
        def unpack_and_remove(datatype, num_elements, file_content):
            datatype = datatype * num_elements
            output = struct.unpack(datatype, file_content[0 : 4 * num_elements])
            file_content = file_content[4 * num_elements :]
            return np.array(output), file_content

        # get quantization and transform (for voxel size)
        response = requests.get(f"{self.path}/info")
        meshes_info = response.json()
        vertex_quantization_bits = meshes_info["vertex_quantization_bits"]
        meshes_transform = meshes_info["transform"]
        meshes_transform += [0, 0, 0, 1]
        meshes_transform = np.reshape(meshes_transform, (4, 4))

        # get index file info
        response = requests.get(f"{self.path}/{id}.index")
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
        response = requests.get(
            f"{self.path}/{id}",
            headers={"range": f"bytes={lod_byte_offset[0]}-{lod_byte_offset[-1]}"},
        )
        lod_bytes = response.content

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

                all_faces = np.concatenate(
                    (all_faces, faces + all_vertices.shape[0]), axis=0
                )
                all_vertices = np.concatenate((all_vertices, vertices), axis=0)
        
        ms = pymeshlab.MeshSet()
        # all_vertices -= np.mean(all_vertices, axis=0)

        m = pymeshlab.Mesh(all_vertices, all_faces)
        ms.add_mesh(m)
        ms.save_current_mesh(f"{id}.ply")
        ms.meshing_remove_duplicate_vertices()
        # ms.meshing_merge_close_vertices(
        #     threshold=pymeshlab.Percentage(
        #         (5 / ms.current_mesh().bounding_box().diagonal()) * 10000,
        #     )
        # )  # meshing_remove_duplicate_vertices()  #
        ms.meshing_remove_duplicate_faces()
        # mesh = trimesh.Trimesh(all_vertices, all_faces, process=False)
        # _ = mesh.export(f"{id}.obj")

        measures = ms.apply_filter("get_topological_measures")
        while measures["number_holes"] != 0:
            ms.meshing_repair_non_manifold_edges()  # sometimes this still has nonmanifold vertices
            ms.meshing_repair_non_manifold_vertices(vertdispratio=0.01)
            # not sure why this doesn't work on first try even though it does in ui
            ms.meshing_close_holes(maxholesize=measures["boundary_edges"] + 1)
            measures = ms.apply_filter("get_topological_measures")
        return ms

    @dask.delayed
    def process_mesh(self, id):
        os.system(f"touch {id}.txt")
        ms = self.get_mesh(id)
        # calculate gneral mesh properties
        metrics = {"id": id}
        measures = ms.apply_filter("get_geometric_measures")
        metrics["volume"] = measures["mesh_volume"]
        metrics["surface_area"] = measures["surface_area"]
        axis_momenta_normalized = measures["axis_momenta"] / np.sum(
            measures["axis_momenta"]
        )
        for axis in range(3):
            metrics[f"axis_momenta_{axis}"] = measures["axis_momenta"][axis]
            metrics[f"axis_momenta_normalized_{axis}"] = axis_momenta_normalized[axis]

        if self.use_skeletons:
            # calculate metrics using navis
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skeleton = sk.skeletonize.by_wavefront(
                    (
                        ms.current_mesh().vertex_matrix(),
                        ms.current_mesh().face_matrix(),
                    ),
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
        os.system(f"rm {id}.txt")
        # return measures["axis_momenta"]
        return metrics
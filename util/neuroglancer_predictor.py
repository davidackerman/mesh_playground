from __future__ import print_function

from functools import partial
import webbrowser

import neuroglancer
import neuroglancer.cli
import copy
import requests

import numpy as np
import pandas as pd


class NeuroglancerPredictor:
    def __init__(
        self,
        dataset,
        organelle,
        class_info=[
            ("class h", "h", "red"),
            ("class j", "j", "gray"),
            ("class k", "k", "blue"),
            ("class l", "l", "magenta"),
        ],
        use_meshes=True,
        selected_segment_ids=None,
        previous_results=None,
    ):
        if not use_meshes:
            assert (
                segmentation_path is not None
            ), "Segmentation path is required when not using meshes"
            vm_path = "https://cellmap-vm1.int.janelia.org"
            segmentation_path = segmentation_path.replace(
                "/nrs/cellmap/", f"{vm_path}/nrs/"
            ).replace("/groups/cellmap/cellmap/", f"{vm_path}/dm11/")
            self.segmentation_path = (
                f"zarr://{segmentation_path}"
                if ".zarr" in segmentation_path
                else f"n5://{segmentation_path}"
            )

        self.dataset = dataset
        self.organelle = organelle
        self.class_info = class_info
        self.use_meshes = use_meshes
        self.selected_segment_ids = selected_segment_ids
        self.previous_results = previous_results
        self.mesh_index_to_class_dict = {}
        if previous_results:
            previous_results_df = pd.read_csv(self.previous_results)
            manually_labeled_class = previous_results_df[
                "Manually Labeled Class"
            ].to_list()
            class_prediction = previous_results_df["Class Prediction"].to_list()

            for mesh_index, manually_labeled_class in enumerate(manually_labeled_class):
                if type(manually_labeled_class) == str:
                    self.mesh_index_to_class_dict[mesh_index] = class_prediction[
                        mesh_index
                    ]

    def setup_neuroglancer(self):
        neuroglancer.set_server_bind_address("0.0.0.0")
        self.viewer = neuroglancer.Viewer()

        if self.use_meshes:
            response = requests.get(
                f"https://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/{self.dataset}/{self.organelle}/multires/segment_properties/info",
            )
            info_file = response.json()
            if not self.selected_segment_ids:
                self.selected_segment_ids = self.all_segment_ids
            self.all_segment_ids = [int(id) for id in info_file["inline"]["ids"]]
            self.mesh_id_to_index_dict = dict(
                zip(self.all_segment_ids, np.arange(len(self.all_segment_ids)))
            )
            with self.viewer.txn() as s:
                s.layers["raw"] = neuroglancer.ImageLayer(  # Single MEsh Layer?
                    source=f"zarr://https://cellmap-vm1.int.janelia.org/nrs/data/{self.dataset}/{self.dataset}.zarr/recon-1/em/fibsem-uint8/",
                )
                s.layers["mesh"] = neuroglancer.SegmentationLayer(  # Single MEsh Layer?
                    source=f"precomputed://https://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/{self.dataset}/{self.organelle}/multires",
                )
                for index, segment_id in enumerate(self.selected_segment_ids):
                    s.layers["mesh"].segments.add(segment_id)
                for class_name, _, _ in self.class_info:
                    s.layers[class_name] = neuroglancer.SegmentationLayer(
                        source=f"precomputed://https://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/{self.dataset}/{self.organelle}/multires",
                    )

                s.layout = "3d"
        else:
            response = requests.get(
                f"https://cellmap-vm1.int.janelia.org/{self.segmentation_path}",
            )
            info_file = response.json()
            self.all_segment_ids = info_file["inline"]["ids"]
            self.mesh_id_to_index_dict = dict(
                zip(self.all_segment_ids, np.arange(len(self.all_segment_ids)))
            )
            with self.viewer.txn() as s:

                s.layers["segmentation"] = neuroglancer.SegmentationLayer(
                    source=f"zarr://https://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/{self.dataset}/{self.organelle}/multires",
                )
                for index, segment_id in enumerate(self.all_segment_ids):
                    self.mesh_id_to_index_dict[int(segment_id)] = index
                    # s.layers["segmentation"].segments.add(segment_id)

                for class_name, _, _ in self.class_info:
                    s.layers[class_name] = neuroglancer.SegmentationLayer(
                        source=f"precomputed://https://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/{self.dataset}/{self.organelle}/multires",
                    )

                s.layout = "3d"

        def manually_label(class_layer, color, s):
            selected_mesh_id = None
            for (
                selected_class_key,
                selected_class_value,
            ) in s.selected_values.iteritems():
                if selected_class_key == "raw":
                    continue
                print(f"{selected_class_key=}")
                selected_mesh_id = selected_class_value.value
                if selected_mesh_id:
                    break

            if selected_mesh_id:
                new_state = copy.deepcopy(self.viewer.state)
                new_state.layers[selected_class_key].segments.remove(selected_mesh_id)
                if (
                    class_layer not in new_state.layers
                ):  # should do it with name since this is a whole object
                    new_state.layers[class_layer] = (
                        neuroglancer.SegmentationLayer(  # Single MEsh Layer?
                            source=f"precomputed://https://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/{self.dataset}/{self.organelle}/multires",
                        )
                    )
                new_state.layers[class_layer].segments.add(selected_mesh_id)

                segment_colors = {}
                for segment_id in new_state.layers[class_layer].segments:
                    segment_colors[segment_id] = color
                new_state.layers[class_layer].segment_colors = segment_colors

                self.viewer.set_state(new_state)

        for name, key, color in self.class_info:
            self.viewer.actions.add(
                f"my-action-{key}", partial(manually_label, name, color)
            )

        with self.viewer.config_state.txn() as s:
            for _, key, _ in self.class_info:
                s.input_event_bindings.viewer[f"key{key}"] = f"my-action-{key}"
            s.input_event_bindings.viewer["keyp"] = "my-action-p"

        url = self.viewer.get_viewer_url()
        # display(IFrame(url, width=1200, height=800)) # to display in jupyter
        webbrowser.open(url)

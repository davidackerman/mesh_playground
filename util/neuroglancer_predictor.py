from __future__ import print_function

import argparse
from functools import partial
import webbrowser

import neuroglancer
import neuroglancer.cli
import copy
import requests

from IPython.display import IFrame
import numpy as np
import socket


class NeuroglancerPredictor:
    def __init__(self, cell):
        self.cell = cell

    def setup_neuroglancer(self):
        neuroglancer.set_server_bind_address(
            bind_address=socket.gethostbyname(socket.gethostname())
        )

        self.viewer = neuroglancer.Viewer()
        response = requests.get(
            f"https://janelia-cosem-datasets.s3.amazonaws.com/{self.cell}/neuroglancer/mesh/mito_seg/segment_properties/info"
        )
        info_file = response.json()
        segment_ids = info_file["inline"]["ids"]
        self.mesh_id_to_index_dict = {}
        with self.viewer.txn() as s:
            s.layers["mesh"] = neuroglancer.SegmentationLayer(  # Single MEsh Layer?
                source=f"precomputed://s3://janelia-cosem-datasets/{self.cell}/neuroglancer/mesh/mito_seg",
            )
            for index, segment_id in enumerate(segment_ids):
                self.mesh_id_to_index_dict[int(segment_id)] = index
                s.layers["mesh"].segments.add(segment_id)
            s.layout = "3d"

        def manually_label(class_key, color, s):

            if s.selected_values["mesh"].value:
                selected_mesh_ids = s.selected_values["mesh"].value
                if s.selected_values["mesh"].value:
                    new_state = copy.deepcopy(self.viewer.state)
                    new_state.layers["mesh"].segments.remove(selected_mesh_ids)
                    class_layer = f"class {class_key}"
                    if (
                        class_layer not in new_state.layers
                    ):  # should do it with name since this is a whole object
                        new_state.layers[
                            class_layer
                        ] = neuroglancer.SegmentationLayer(  # Single MEsh Layer?
                            source=f"precomputed://s3://janelia-cosem-datasets/{self.cell}/neuroglancer/mesh/mito_seg"
                        )
                    new_state.layers[class_layer].segments.add(selected_mesh_ids)

                    segment_colors = {}
                    for segment_id in new_state.layers[class_layer].segments:
                        segment_colors[segment_id] = color
                    new_state.layers[class_layer].segment_colors = segment_colors

                    self.viewer.set_state(new_state)

        self.viewer.actions.add("my-action-h", partial(manually_label, "h", "red"))
        self.viewer.actions.add("my-action-j", partial(manually_label, "j", "gray"))
        self.viewer.actions.add("my-action-k", partial(manually_label, "k", "blue"))
        self.viewer.actions.add("my-action-l", partial(manually_label, "l", "magenta"))

        with self.viewer.config_state.txn() as s:
            s.input_event_bindings.viewer["keyh"] = "my-action-h"
            s.input_event_bindings.viewer["keyj"] = "my-action-j"
            s.input_event_bindings.viewer["keyk"] = "my-action-k"
            s.input_event_bindings.viewer["keyl"] = "my-action-l"
            s.input_event_bindings.viewer["keyp"] = "my-action-p"

        url = self.viewer.get_viewer_url()
        # display(IFrame(url, width=1200, height=800)) # to display in jupyter
        webbrowser.open(url)

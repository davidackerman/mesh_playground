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
    def __init__(self, cell,
                 organelle, 
                 class_info=[("class h", "h", "red"), 
                      ("class j", "j", "gray"), 
                      ("class k", "k", "blue"), 
                      ("class l", "l", "magenta")]):
        self.cell = cell
        self.organelle = organelle
        self.class_info = class_info

    def setup_neuroglancer(self):
        neuroglancer.set_server_bind_address("0.0.0.0")
        self.viewer = neuroglancer.Viewer()
        response = requests.get(
            # f"https://janelia-cosem-datasets.s3.amazonaws.com/{self.cell}/neuroglancer/mesh/mito_seg/segment_properties/info"
            f"https://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/{self.cell}/{self.organelle}/multires/segment_properties/info",
        )
        info_file = response.json()
        self.segment_ids = info_file["inline"]["ids"]
        self.mesh_id_to_index_dict = {}
        with self.viewer.txn() as s:
            # s.layers["raw"] = neuroglancer.ImageLayer(  # Single MEsh Layer?
            #     # source=f"precomputed://s3://janelia-cosem-datasets/{self.cell}/neuroglancer/mesh/mito_seg",
            #     source=f"n5://https://cellmap-vm1.int.janelia.org/nrs/data/{self.cell}/{self.cell}.n5/em/fibsem-uint8/",
            # )
            s.layers["mesh"] = neuroglancer.SegmentationLayer(  # Single MEsh Layer?
                # source=f"precomputed://s3://janelia-cosem-datasets/{self.cell}/neuroglancer/mesh/mito_seg",
                source=f"precomputed://https://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/{self.cell}/{self.organelle}/multires",
            )
            for index, segment_id in enumerate(self.segment_ids):
                self.mesh_id_to_index_dict[int(segment_id)] = index
                s.layers["mesh"].segments.add(segment_id)
            
            for class_name,_,_ in self.class_info:
                s.layers[class_name] = neuroglancer.SegmentationLayer(  # Single MEsh Layer?
                    source=f"precomputed://https://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/{self.cell}/{self.organelle}/multires",
                )

            s.layout = "3d"

        def manually_label(class_layer, color, s):
            selected_mesh_id = None
            for selected_class_key,selected_class_value in s.selected_values.iteritems():
                selected_mesh_id = selected_class_value.value
                if selected_mesh_id:
                    break
            
            if selected_mesh_id:
                new_state = copy.deepcopy(self.viewer.state)
                new_state.layers[selected_class_key].segments.remove(selected_mesh_id)
                if (
                    class_layer not in new_state.layers
                ):  # should do it with name since this is a whole object
                    new_state.layers[
                        class_layer
                    ] = neuroglancer.SegmentationLayer(  # Single MEsh Layer?
                        # source=f"precomputed://s3://janelia-cosem-datasets/{self.cell}/neuroglancer/mesh/mito_seg"
                        source=f"precomputed://https://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/{self.cell}/{self.organelle}/multires"
                    )
                new_state.layers[class_layer].segments.add(selected_mesh_id)

                segment_colors = {}
                for segment_id in new_state.layers[class_layer].segments:
                    segment_colors[segment_id] = color
                new_state.layers[class_layer].segment_colors = segment_colors

                self.viewer.set_state(new_state)
    
        for (name, key, color) in self.class_info:
            self.viewer.actions.add(f"my-action-{key}", partial(manually_label, name, color))

        with self.viewer.config_state.txn() as s:
            for (_, key, _) in self.class_info:
                s.input_event_bindings.viewer[f"key{key}"] = f"my-action-{key}"
            s.input_event_bindings.viewer["keyp"] = "my-action-p"

        url = self.viewer.get_viewer_url()
        # display(IFrame(url, width=1200, height=800)) # to display in jupyter
        webbrowser.open(url)

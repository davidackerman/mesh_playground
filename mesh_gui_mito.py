"""
Glooey widget example. Only runs with python>=3.6 (because of Glooey).
"""

import pathlib
from re import S
import time
import glooey
import numpy as np
import pyglet

pyglet.options["debug_gl"] = False

import trimesh
import trimesh.viewer
import networkx as nx
from sklearn import svm, ensemble, preprocessing
from sklearn.neural_network import MLPClassifier

import numba
from numba import jit
from numba.typed import Dict
import pymeshlab
from scipy.sparse import csc_matrix


here = pathlib.Path(__file__).resolve().parent
import colorsys


class Application:

    """
    Example application that includes moving camera, scene and image update.
    """

    def __init__(self):
        from util.mesh import MeshProcessor
        from tqdm import tqdm
        import numpy as np
        import dask
        from dask.distributed import Client, progress
        import pandas as pd

        client = Client(threads_per_worker=2, n_workers=1)
        client.cluster.scale(8)
        print(client.dashboard_link)
        mp = MeshProcessor(
            path="https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/neuroglancer/mesh/mito_seg",
            lod=2,
            min_branch_length=100,
            use_skeletons=True,
        )
        # momenta = np.zeros((421,3),dtype=np.float32)
        lazy_results = []
        for i in range(1, 422, 1):  # range(1,87077,100): #
            lazy_results.append(mp.process_mesh(id=i))
        results = dask.compute(*lazy_results)
        df = pd.DataFrame.from_records(results)
        client.shutdown()
        self.metrics = df[
            [
                "longest_path",
                "num_fragments",
                "volume",
                "principal_inertia_component_normalized_0",
                "principal_inertia_component_normalized_1",
                "principal_inertia_component_normalized_2",
            ]
        ].to_numpy()

        self.next_display_type_idx = 0
        self.width, self.height = 960, 960
        self.mouse_pos = 0, 0

        window = self._create_window(width=self.width, height=self.height)

        self.key_pressed = None
        self.meshes_to_manual_class_dict = {}
        self.meshes_to_class_dict = {}
        self.class_colors = np.array(
            [
                [127, 127, 127],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 255],
                [0, 255, 255],
                [255, 255, 0],
            ],
        )

        gui = glooey.Gui(window)

        hbox = glooey.HBox()
        self.padding = 0
        hbox.set_padding(self.padding)

        # scene widget for changing camera location

        scene = trimesh.Scene()
        self.selected_mesh = None
        self.selected_id = None
        self.all_meshes = []
        self.mesh_face_ids = []
        self.mesh_ids = []
        for idx, row in df.iterrows():
            mesh = row["mesh"]
            id = int(row["id"])
            mesh.vertices /= 1000
            self.all_meshes.append(mesh)
            mesh.visual.face_colors = self.lighten_color(
                trimesh.visual.color.random_color(), 0.1
            )
            # scene.add_geometry(mesh, geom_name=f"mesh_{id}")
            self.mesh_ids.append(id)
            self.mesh_face_ids.extend([id] * len(mesh.faces))

        self.all_meshes = trimesh.util.concatenate(self.all_meshes)
        scene.add_geometry(self.all_meshes, geom_name="all_meshes")
        # self.widgets = []
        self.widget = trimesh.viewer.SceneWidget(scene, smooth=True)
        self.widget._background = [0, 0, 0, 255]
        self.widget.scene.camera._fov = [45.0, 45.0]

        hbox.add(self.widget)

        gui.add(hbox)

        pyglet.app.run()

    def lighten_color(self, rgb, fraction):
        r = rgb[0] / 255.0
        g = rgb[1] / 255.0
        b = rgb[2] / 255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = l + (1 - l) * fraction
        # s = s - 0.5 * s
        rgb_new = colorsys.hls_to_rgb(h, l, s)
        return rgb_new

    def cursor_triangle(self):
        x, y = self.mouse_pos
        origins, vectors, _ = self.widget.scene.camera_rays()
        resolution = self.widget.scene.camera.resolution
        left = self.widget.rect.left
        bottom = self.widget.rect.bottom
        x_in_scene, y_in_scene = x - left - self.padding, y - bottom - self.padding
        idx = x_in_scene * resolution[1] + ((resolution[1] - 1) - y_in_scene)
        current_origin, current_vector = origins[idx], vectors[idx]
        _, _, triangle_index = self.all_meshes.ray.intersects_location(
            [current_origin], [current_vector], multiple_hits=False
        )
        if triangle_index.size > 0:
            return triangle_index[0]
        else:
            return None

    def get_current_widget(self):
        x, y = self.mouse_pos
        for current_widget in self.widget:
            left = current_widget.rect.left
            bottom = current_widget.rect.bottom
            width = current_widget.rect.width
            height = current_widget.rect.height
            if (left < x <= left + width) and (bottom < y <= bottom + height):
                break
        self.current_widget = current_widget

    def recenter(self):
        triangle_index = self.cursor_triangle()
        previous_centroid = self.widget.scene.centroid
        if "bounding_box_needed_for_centering" not in self.widget.scene.geometry:
            geom = trimesh.path.creation.box_outline((1000, 1000, 1000))
            self.widget.scene.add_geometry(
                geom, geom_name="bounding_box_needed_for_centering"
            )
            self.previous_offset = 0, 0, 0
        geom = self.widget.scene.geometry["bounding_box_needed_for_centering"]
        geom.vertices -= self.previous_offset

        center = self.all_meshes.triangles_center[triangle_index]
        geom.vertices += center
        self.previous_offset = center

        self.widget.scene.camera_transform[:3, 3] += center - previous_centroid
        self.widget._initial_camera_transform = self.widget.scene.camera_transform

        self.widget.reset_view()

    def fit(self):
        # self.classifier = svm.SVC()
        # self.classifier = ensemble.RandomForestClassifier()
        self.classifier = MLPClassifier(alpha=1, max_iter=1000)
        classes = list(self.meshes_to_manual_class_dict.values())

        mesh_ids = list(self.meshes_to_manual_class_dict.keys())
        mesh_indices = [self.mesh_ids.index(mesh_id) for mesh_id in mesh_ids]
        data = self.metrics[mesh_indices, :]
        self.scaler = preprocessing.StandardScaler().fit(data)
        data_transformed = self.scaler.transform(data)
        self.classifier.fit(data_transformed, classes)

    def predict(self):
        test_transformed = self.scaler.transform(self.metrics)
        class_predictions = self.classifier.predict(test_transformed)
        for idx, class_prediction in enumerate(class_predictions):
            if self.mesh_ids[idx] not in self.meshes_to_manual_class_dict:
                mesh = self.widget.scene.geometry[f"mesh_{self.mesh_ids[idx]}"]
                class_color = self.class_colors[class_prediction]
                mesh.visual.face_colors = self.lighten_color(class_color, 0.6)
        self.widget._draw()

    def _create_window(self, width, height):
        try:
            # config = pyglet.gl.Config(
            #     sample_buffers=1, samples=4, depth_size=24, double_buffer=True
            # )
            config = pyglet.gl.Config(
                sample_buffers=1, samples=4, depth_size=24, double_buffer=True
            )
            window = pyglet.window.Window(config=config, width=width, height=height)
        except pyglet.window.NoSuchConfigException:
            config = pyglet.gl.Config(double_buffer=True)
            window = pyglet.window.Window(config=config, width=width, height=height)
        self.display_type_label = pyglet.text.Label(
            "Original Mesh",
            font_size=18,
            x=window.width // 2,
            y=window.height // 10,
            anchor_x="center",
            anchor_y="center",
            color=(0, 0, 0, 255),
        )
        self.center_label = pyglet.text.Label(
            "",
            font_size=18,
            x=window.width // 2,
            y=window.height * 9.5 / 10,
            anchor_x="center",
            anchor_y="center",
        )

        # @window.event
        # def on_draw():
        # self.display_type_label.draw()

        @window.event
        def on_key_release(symbol, modifiers):
            # if symbol in  [pyglet.window.key._1, pyglet.window.key._2, pyglet.window.key._3, pyglet.window.key._4, pyglet.window.key._5, pyglet.window.key._6, pyglet.window.key._7]:
            self.key_pressed = None

        @window.event
        def on_key_press(symbol, modifiers):
            if modifiers == 0:
                if symbol == pyglet.window.key._1:
                    self.key_pressed = 1
                if symbol == pyglet.window.key._2:
                    self.key_pressed = 2
                if symbol == pyglet.window.key._3:
                    self.key_pressed = 3
                if symbol == pyglet.window.key._4:
                    self.key_pressed = 4
                if symbol == pyglet.window.key._5:
                    self.key_pressed = 5
                if symbol == pyglet.window.key.Q:
                    window.close()
                if symbol == pyglet.window.key.C:
                    self.recenter()
                if symbol == pyglet.window.key.P:
                    self.center_label.text = "predicting"
                    self.fit()
                    self.predict()
                    self.center_label.text = "predicted"
                if symbol == pyglet.window.key.T:
                    if self.next_display_type_idx == len(self.metrics):
                        self.next_display_type_idx = 0
                        self.display_type_label.text = "Original Mesh"
                        self.widget.mesh.visual.vertex_colors = self.mesh_colors
                    else:
                        display_type = list(self.metrics.keys())[
                            self.next_display_type_idx
                        ]
                        self.display_type_label.text = display_type
                        metric = self.metrics[display_type]
                        self.widgets[
                            0
                        ].mesh.visual.face_colors = trimesh.visual.interpolate(
                            metric.clip(
                                np.percentile(metric, 10), np.percentile(metric, 90)
                            ),
                            color_map="viridis",
                        )
                        self.next_display_type_idx += 1
                    self.widget._draw()

                if self.key_pressed and self.selected_id:
                    # label a mesh
                    self.meshes_to_manual_class_dict[
                        self.selected_id
                    ] = self.key_pressed
                    group_color = self.class_colors[self.key_pressed, :]
                    self.previous_color = group_color
                    self.selected_mesh.visual.face_colors = self.lighten_color(
                        group_color, 0.6
                    )
                    self.widget._draw()

        @window.event
        def on_mouse_motion(x, y, dx, dy):
            self.mouse_pos = x, y
            # if self.key_pressed:
            t = time.time()
            cursor_triangle = self.cursor_triangle()
            print("a", time.time() - t)
            t = time.time()
            if cursor_triangle:
                cursor_id = self.mesh_face_ids[cursor_triangle]
                if cursor_id != self.selected_id:
                    if self.selected_id:
                        self.selected_mesh.visual.face_colors = self.previous_color
                        # select new mesh
                    self.selected_mesh = self.widget.scene.geometry[f"mesh_{cursor_id}"]
                    self.selected_id = cursor_id
                    self.previous_color = self.selected_mesh.visual.face_colors[
                        0, 0:3
                    ].astype(np.float16)

                    self.selected_mesh.visual.face_colors = self.lighten_color(
                        self.previous_color, 0.6
                    )
                    print("b", time.time() - t)
                    t = time.time()
                    self.widget._draw()
                    print("c", time.time() - t)
            else:
                if self.selected_id:
                    self.selected_mesh.visual.face_colors = self.previous_color
                    self.selected_mesh = None
                    self.selected_id = None
                    self.widget._draw()

            # redraw = False
            # current_class = self.key_pressed
            # geom_name = f"{triangle_index}"
            # if triangle_index not in self.triangle_indices_to_class_dict:
            #     redraw = True
            #     submesh = self.widget.mesh_scaled.submesh([[triangle_index]])[0]
            #     submesh.visual.face_colors = self.class_colors[current_class]
            #     self.widget.scene.add_geometry(submesh, geom_name=geom_name)
            # else:
            #     redraw = False
            #     previous_class = self.triangle_indices_to_class_dict[triangle_index]
            #     if previous_class != current_class:
            #         self.widget.scene.geometry[
            #             geom_name
            #         ].visual.face_colors = self.class_colors[current_class]
            #         redraw = True

            # if redraw:
            #     self.triangle_indices_to_class_dict[triangle_index] = current_class
            #     for v in self.widget.mesh.faces[triangle_index]:
            #         self.vertex_indices_to_class_dict[v] = current_class
            #     self.widget._draw()

        return window


if __name__ == "__main__":
    np.random.seed(0)
    Application()

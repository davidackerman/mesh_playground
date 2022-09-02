"""
Glooey widget example. Only runs with python>=3.6 (because of Glooey).
"""

import pathlib
import time
import glooey
import numpy as np
import pyglet

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


def average_over_one_ring(mesh, metrics):
    if type(metrics) is not list:
        metrics = tuple(metrics)
    metrics = np.column_stack(metrics)
    g = nx.from_edgelist(mesh.edges_unique)
    # t = time.time()
    # temp = dict(nx.all_pairs_shortest_path_length(g, cutoff=5))
    # print(time.time() - t)
    avgs = np.array(
        [
            np.mean(metrics[list(g[i].keys())][:], axis=0)
            for i in range(len(mesh.vertices))
        ]
    )

    return [avgs[:, col] for col in range(avgs.shape[1])]


def average_over_n_ring(mesh, c, n):
    g = nx.from_edgelist(mesh.edges_unique)
    one_rings = [list(g[i].keys()) for i in range(len(mesh.vertices))]

    avg = [0] * len(one_rings)
    for vertex_id in range(len(one_rings)):

        current_ring = one_rings[vertex_id]
        n_ring_vertex_ids = current_ring
        for ring in range(n - 1):
            # can speed up so not rechecking
            next_ring = []
            for i in current_ring:
                next_ring.extend(one_rings[i])
            current_ring = np.unique(next_ring)
            n_ring_vertex_ids.extend(current_ring)
        avg[vertex_id] = np.mean(c[np.unique(n_ring_vertex_ids)])

    return np.array(avg)


class Application:

    """
    Example application that includes moving camera, scene and image update.
    """

    def __init__(self):
        # geom = trimesh.load(str(here / "content/MPI-FAUST/meshes/tr_reg_000.ply"))
        ms = pymeshlab.MeshSet()
        # ms.load_new_mesh("/Users/ackermand/Documents/Downloads/410_roi1.obj")
        ms.load_new_mesh("./content/MPI-FAUST/meshes/tr_reg_000.ply")
        mesh = ms.current_mesh()
        t = time.time()
        ms.meshing_repair_non_manifold_edges()
        self.metrics = {}
        ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=0)

        smoothing_iterations = [0, 1, 5, 10]
        self.laplacian_smooth(ms, mesh, "mean_curvature", smoothing_iterations)

        print("mc", time.time() - t)
        t = time.time()
        ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=1)
        self.laplacian_smooth(ms, mesh, "gaussian_curvature", smoothing_iterations)
        print("gc", time.time() - t)
        t = time.time()

        ms.compute_scalar_by_shape_diameter_function_per_vertex()
        self.metrics["thickness"] = mesh.vertex_scalar_array()
        self.laplacian_smooth(ms, mesh, "thickness_curvature", smoothing_iterations)

        print("th", time.time() - t)
        t = time.time()

        ms.compute_scalar_by_volumetric_obscurance()
        ms.apply_scalar_smoothing_per_vertex()
        # self.metrics["obscurance"] = mesh.vertex_scalar_array()

        self.next_display_type_idx = 0
        # create window with padding
        self.width, self.height = 480 * 2, 480
        self.mouse_pos = 0, 0

        window = self._create_window(width=self.width, height=self.height)

        self.key_pressed = None
        self.triangle_indices_to_group_dict = {}
        self.vertex_indices_to_group_dict = {}
        self.triangle_indices_by_group = [set(), set(), set()]
        self.group_colors = [
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 0, 0, 1.0),
            (0.0, 1.0, 0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0, 1.0, 1.0),
            (0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 0.0, 1.0),
        ]

        gui = glooey.Gui(window)

        hbox = glooey.HBox()
        self.padding = 0
        hbox.set_padding(self.padding)

        # scene widget for changing camera location

        scene = trimesh.Scene()

        geom = trimesh.Trimesh(vertices=mesh.vertex_matrix(), faces=mesh.face_matrix())

        print(len(geom.vertices))
        geom.vertices = (geom.vertices - np.min(geom.vertices, axis=0)) / (
            np.amax(geom.vertices) - np.min(geom.vertices, axis=0)
        )
        obscurance = mesh.vertex_scalar_array()

        self.mesh_colors = trimesh.visual.interpolate(obscurance, color_map="gray")
        geom.visual.vertex_colors = self.mesh_colors

        scene.add_geometry(geom, geom_name="mesh")
        # scene.lights = trimesh.scene.lighting.autolight(scene)[0]
        # print(scene.lights)
        t = time.time()
        # self.mesh = scene.geometry["mesh"]

        self.widgets = []
        self.widgets.append(trimesh.viewer.SceneWidget(scene, smooth=False))
        self.widgets[0].scene.camera._fov = [45.0, 45.0]
        self.widgets[0].mesh = geom
        self.widgets[0].mesh_scaled = self.widgets[0].mesh.copy()
        self.widgets[0].mesh_scaled.vertices += (
            self.widgets[0].mesh_scaled.vertex_normals * 0.00001
        )
        self.widgets[0].triangle_group_assignments = [] * len(
            self.widgets[0].mesh_scaled.faces
        )

        hbox.add(self.widgets[0])

        # scene widget for changing scene
        mesh_copy = self.widgets[0].mesh.copy()
        scene = trimesh.Scene()
        scene.add_geometry(mesh_copy, geom_name="mesh")
        self.widgets.append(trimesh.viewer.SceneWidget(scene, smooth=False))
        self.widgets[1].mesh = mesh_copy
        self.widgets[1].mesh.visual.vertex_colors = self.mesh_colors
        hbox.add(self.widgets[1])

        gui.add(hbox)

        self.average_metrics_over_faces()

        pyglet.app.run()

    def laplacian_smooth(self, ms, mesh, metric, smoothing_iterations):
        for iteration in range(max(smoothing_iterations) + 1):
            if iteration in smoothing_iterations:
                self.metrics[f"{metric}_{iteration}"] = mesh.vertex_scalar_array()
            ms.apply_scalar_smoothing_per_vertex()

    def average_metrics_over_faces(self):
        faces = self.widgets[0].mesh.faces.flatten()
        for metric_name in self.metrics:
            metric = self.metrics[metric_name]
            self.metrics[metric_name] = np.mean(metric[faces].reshape((-1, 3)), axis=1)

    def cursor_triangle(self):
        self.get_current_widget()
        x, y = self.mouse_pos
        origins, vectors, _ = self.current_widget.scene.camera_rays()
        resolution = self.current_widget.scene.camera.resolution
        left = self.current_widget.rect.left
        bottom = self.current_widget.rect.bottom
        x_in_scene, y_in_scene = x - left - self.padding, y - bottom - self.padding
        idx = x_in_scene * resolution[1] + ((resolution[1] - 1) - y_in_scene)
        current_origin, current_vector = origins[idx], vectors[idx]
        _, _, triangle_index = self.current_widget.mesh.ray.intersects_location(
            [current_origin], [current_vector], multiple_hits=False
        )
        return triangle_index[0]

    def get_current_widget(self):
        x, y = self.mouse_pos
        for current_widget in self.widgets:
            left = current_widget.rect.left
            bottom = current_widget.rect.bottom
            width = current_widget.rect.width
            height = current_widget.rect.height
            if (left < x <= left + width) and (bottom < y <= bottom + height):
                break
        self.current_widget = current_widget

    def recenter(self):
        triangle_index = self.cursor_triangle()
        current_widget = self.current_widget
        previous_centroid = current_widget.scene.centroid
        if "bounding_box_needed_for_centering" not in current_widget.scene.geometry:
            geom = trimesh.path.creation.box_outline((1000, 1000, 1000))
            current_widget.scene.add_geometry(
                geom, geom_name="bounding_box_needed_for_centering"
            )
            current_widget.previous_offset = 0, 0, 0
        geom = current_widget.scene.geometry["bounding_box_needed_for_centering"]
        geom.vertices -= current_widget.previous_offset

        center = current_widget.mesh.triangles_center[triangle_index]
        geom.vertices += center
        current_widget.previous_offset = center

        current_widget.scene.camera_transform[:3, 3] += center - previous_centroid
        current_widget._initial_camera_transform = current_widget.scene.camera_transform

        current_widget.reset_view()

    def fit(self):
        # self.classifier = svm.SVC()
        # self.classifier = ensemble.RandomForestClassifier()
        self.classifier = MLPClassifier(alpha=1, max_iter=1000)
        faces = list(self.triangle_indices_to_group_dict.keys())
        groups = list(self.triangle_indices_to_group_dict.values())
        data = [metric[faces] for metric in self.metrics.values()]
        data = list(zip(*data))
        self.scaler = preprocessing.StandardScaler().fit(data)
        data_transformed = self.scaler.transform(data)
        self.classifier.fit(data_transformed, groups)

    def predict(self):
        test = list(zip(*self.metrics.values()))
        test_transformed = self.scaler.transform(test)
        group_predictions = self.classifier.predict(test_transformed)
        colors = [self.group_colors[group] for group in group_predictions]

        self.widgets[1].mesh.visual.face_colors = colors  # vertex_colors
        self.widgets[1]._draw()

    def _create_window(self, width, height):
        try:
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
            x=window.width // 4,
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

        @window.event
        def on_draw():
            self.display_type_label.draw()

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
                        self.widgets[0].mesh.visual.vertex_colors = self.mesh_colors
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
                    self.widgets[0]._draw()

        # )

        @window.event
        def on_mouse_motion(x, y, dx, dy):
            self.mouse_pos = x, y
            if self.key_pressed:
                triangle_index = self.cursor_triangle()
                redraw = False
                current_group = self.key_pressed
                geom_name = f"{triangle_index}"
                if triangle_index not in self.triangle_indices_to_group_dict:
                    redraw = True
                    submesh = self.widgets[0].mesh_scaled.submesh([[triangle_index]])[0]
                    submesh.visual.face_colors = self.group_colors[current_group]
                    self.widgets[0].scene.add_geometry(submesh, geom_name=geom_name)
                else:
                    redraw = False
                    previous_group = self.triangle_indices_to_group_dict[triangle_index]
                    if previous_group != current_group:
                        self.widgets[0].scene.geometry[
                            geom_name
                        ].visual.face_colors = self.group_colors[current_group]
                        redraw = True

                if redraw:
                    self.triangle_indices_to_group_dict[triangle_index] = current_group
                    for v in self.widgets[0].mesh.faces[triangle_index]:
                        self.vertex_indices_to_group_dict[v] = current_group
                    self.widgets[0]._draw()

        # @window.event
        # def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        #     x_prev = x - dx
        #     y_prev = y - dy
        #     left, bottom = self.widgets[0].rect.left, self.widgets[0].rect.bottom
        #     width, height = (
        #         self.widgets[0].rect.width,
        #         self.widgets[0].rect.height,
        #     )
        #     if not (left < x_prev <= left + width) or not (
        #         bottom < y_prev <= bottom + height
        #     ):
        #         print("w1", x, x_prev, y, y_prev)
        #     left, bottom = self.widgets[1].rect.left, self.widgets[1].rect.bottom
        #     width, height = (
        #         self.widgets[1].rect.width,
        #         self.widgets[1].rect.height,
        #     )
        #     if not (left < x_prev <= left + width) or not (
        #         bottom < y_prev <= bottom + height
        #     ):
        #         print("w2", x, x_prev, y, y_prev)

        #     print(
        #         self.widgets[0].view["ball"]._pdown,
        #         self.widgets[0].view["ball"]._pose,
        #         self.widgets[0].view["ball"]._target,
        #     )

        return window


if __name__ == "__main__":
    np.random.seed(0)
    Application()

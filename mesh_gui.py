"""
Glooey widget example. Only runs with python>=3.6 (because of Glooey).
"""

import io
import pathlib
import time
import glooey
import numpy as np
import pyglet

import trimesh
import trimesh.viewer
import trimesh.transformations as tf
import PIL.Image
import networkx as nx
from sklearn import svm, ensemble, preprocessing

import numba
from numba import jit
from numba.typed import Dict
import pymeshlab

here = pathlib.Path(__file__).resolve().parent


def average_over_one_ring(mesh, metrics):
    if type(metrics) is not list:
        metrics = [metrics]
    g = nx.from_edgelist(mesh.edges_unique)
    one_rings = [list(g[i].keys()) for i in range(len(mesh.vertices))]

    avgs = [np.zeros_like(metric) for metric in metrics]
    for current_avg, current_metric in zip(avgs, metrics):
        for vertex_id in range(len(mesh.vertices)):
            current_avg[vertex_id] = np.mean(current_metric[one_rings[vertex_id]])

    return avgs


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


def my_discrete_mean_curvature_measure(mesh):
    """Calculate discrete mean curvature of mesh using one-ring neighborhood."""

    # one-rings (immediate neighbors of) each vertex
    g = nx.from_edgelist(mesh.edges_unique)

    # cotangents of angles and store in dictionary based on corresponding vertex and face
    face_angles = mesh.face_angles_sparse
    keys = tuple(zip(face_angles.row, face_angles.col))
    cotangents = dict(zip(keys, 1 / np.tan(face_angles.data)))

    # discrete Laplace-Beltrami contribution of the shared edge of adjacent faces:
    #        /*\
    #       / * \
    #      /  *  \
    #    vi___*___vj
    #
    # store results in dictionary with vertex ids as keys
    fa = mesh.face_adjacency
    fae = mesh.face_adjacency_edges
    keys = tuple(zip(fae[:, 0], fae[:, 1]))
    cotangent_sums = np.array(
        [
            cotangents[(v[0], fa[i][0])] + cotangents[(v[1], fa[i][1])]
            for i, v in enumerate(mesh.face_adjacency_unshared)
        ]
    )
    edge_measure = dict(
        zip(
            keys,
            (mesh.vertices[fae[:, 1]] - mesh.vertices[fae[:, 0]])
            * cotangent_sums[:, None],
        )
    )

    # calculate mean curvature using one-ring
    mean_curv = [0] * len(mesh.vertices)
    for vertex_id, face_ids in enumerate(mesh.vertex_faces):
        face_ids = face_ids[face_ids != -1]  # faces associated with vertex_id
        one_ring = list(g[vertex_id].keys())
        delta_s = 0

        for one_ring_vertex_id in one_ring:
            if (vertex_id, one_ring_vertex_id) in edge_measure:
                delta_s += edge_measure[(vertex_id, one_ring_vertex_id)]
            elif (one_ring_vertex_id, vertex_id) in edge_measure:
                delta_s -= edge_measure[(one_ring_vertex_id, vertex_id)]

        delta_s *= 1 / (
            2 * np.sum(mesh.area_faces[face_ids]) / 3
        )  # use 1/3 of the areas
        mean_curv[vertex_id] = 0.5 * np.linalg.norm(delta_s)

    return np.array(mean_curv)


def old_my_discrete_mean_curvature_measure(mesh):
    """Calculate discrete mean curvature of mesh using one-ring neighborhood."""

    # one-rings (immediate neighbors of) each vertex
    t = time.time()
    g = nx.from_edgelist(mesh.edges_unique)
    one_rings = [list(g[i].keys()) for i in range(len(mesh.vertices))]
    print("inside", time.time() - t)
    t = time.time()

    # cotangents of angles and store in dictionary based on corresponding vertex and face

    print("inside 3", time.time() - t)
    t = time.time()
    face_angles = mesh.face_angles_sparse
    cotangents = {
        f"{vertex},{face}": 1 / np.tan(angle)
        for vertex, face, angle in zip(
            face_angles.row, face_angles.col, face_angles.data
        )
    }
    print("inside 4", time.time() - t)
    t = time.time()
    # discrete Laplace-Beltrami contribution of the shared edge of adjacent faces:
    #        /*\
    #       / * \
    #      /  *  \
    #    vi___*___vj
    #
    # store results in dictionary with vertex ids as keys
    fa = mesh.face_adjacency
    fae = mesh.face_adjacency_edges
    edge_measure = {
        f"{fae[i][0]},{fae[i][1]}": (
            mesh.vertices[fae[i][1]] - mesh.vertices[fae[i][0]]
        )
        * (cotangents[f"{v[0]},{fa[i][0]}"] + cotangents[f"{v[1]},{fa[i][1]}"])
        for i, v in enumerate(mesh.face_adjacency_unshared)
    }
    # calculate mean curvature using one-ring
    mean_curv = [0] * len(mesh.vertices)
    for vertex_id, face_ids in enumerate(mesh.vertex_faces):
        face_ids = face_ids[face_ids != -1]  # faces associated with vertex_id
        one_ring = one_rings[vertex_id]
        delta_s = 0

        for one_ring_vertex_id in one_ring:
            if f"{vertex_id},{one_ring_vertex_id}" in edge_measure:
                delta_s += edge_measure[f"{vertex_id},{one_ring_vertex_id}"]
            elif f"{one_ring_vertex_id},{vertex_id}" in edge_measure:
                delta_s -= edge_measure[f"{one_ring_vertex_id},{vertex_id}"]

        delta_s *= 1 / (
            2 * np.sum(mesh.area_faces[face_ids]) / 3
        )  # use 1/3 of the areas
        mean_curv[vertex_id] = 0.5 * np.linalg.norm(delta_s)

    return np.array(mean_curv)


def my_discrete_gaussian_curvature_measure(mesh):
    """
    Return the discrete gaussian curvature measure of a sphere centered
    at a point as detailed in 'Restricted Delaunay triangulations and normal
    cycle', Cohen-Steiner and Morvan.
    Parameters
    ----------
    points : (n,3) float, list of points in space
    radius : float, the sphere radius
    Returns
    --------
    gaussian_curvature: (n,) float, discrete gaussian curvature measure.
    """

    g = nx.from_edgelist(mesh.edges_unique)
    # nearest = mesh.kdtree.query_ball_point(points, radius)
    one_ring = [list(g[i].keys()) for i in range(len(mesh.vertices))]

    # FACTOR OF 3 since using 1/3 area?
    gauss_curv = [
        3
        * mesh.vertex_defects[vertex]
        / mesh.area_faces[
            mesh.vertex_faces[vertex][mesh.vertex_faces[vertex] != -1]
        ].sum()
        for vertex in range(len(mesh.vertices))
    ]

    return np.asarray(gauss_curv)


class Application:

    """
    Example application that includes moving camera, scene and image update.
    """

    def __init__(self):
        # geom = trimesh.load(str(here / "content/MPI-FAUST/meshes/tr_reg_000.ply"))
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh("/Users/ackermand/Documents/Downloads/410_roi1.obj")
        # ms.load_new_mesh("./content/MPI-FAUST/meshes/tr_reg_050.ply")
        mesh = ms.current_mesh()
        t = time.time()
        ms.meshing_repair_non_manifold_edges()
        ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=0)

        mean_curvature = mesh.vertex_scalar_array()

        print("mc", time.time() - t)
        t = time.time()
        ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=1)
        gaussian_curvature = mesh.vertex_scalar_array()
        print("gc", time.time() - t)
        t = time.time()

        ms.compute_scalar_by_shape_diameter_function_per_vertex()
        thickness = mesh.vertex_scalar_array()
        print("th", time.time() - t)
        t = time.time()

        ms.compute_scalar_by_volumetric_obscurance()
        obscurance = mesh.vertex_scalar_array()

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
        print(len(gaussian_curvature))
        geom.vertices = (geom.vertices - np.min(geom.vertices, axis=0)) / (
            np.amax(geom.vertices) - np.min(geom.vertices, axis=0)
        )
        print(np.max(geom.vertices, axis=0), np.min(geom.vertices, axis=0))
        obs = mesh.vertex_scalar_array()
        vc = []

        obs = (obs - np.min(obs)) / (np.max(obs) - np.min(obs))
        for o in obs:
            vc.append([o, o, o, 1.0])
        geom.visual.vertex_colors = vc

        scene.add_geometry(geom, geom_name="original_mesh")
        # scene.lights = trimesh.scene.lighting.autolight(scene)[0]
        # print(scene.lights)
        self.original_mesh = scene.geometry["original_mesh"]
        (
            self.mean_curvature,
            self.gaussian_curvature,
            self.thickness,
            self.obscurance,
        ) = average_over_one_ring(
            self.original_mesh,
            [mean_curvature, gaussian_curvature, thickness, obscurance],
        )
        print("mca", time.time() - t)

        self.original_mesh_scaled = self.original_mesh.copy()
        self.original_mesh_scaled.vertices += (
            self.original_mesh_scaled.vertex_normals * 0.0001
        )
        self.triangle_group_assignments = [] * len(self.original_mesh_scaled.faces)
        self.scene_widget1 = trimesh.viewer.SceneWidget(scene)
        self.scene_widget1.scene.camera._fov = [45.0, 45.0]

        hbox.add(self.scene_widget1)

        # scene widget for changing scene
        scene = trimesh.Scene()

        self.original_mesh_window2 = self.original_mesh.copy()
        self.original_mesh_window2.visual.vertex_colors = [0.5, 0.5, 0.5, 1]
        scene.add_geometry(self.original_mesh_window2, geom_name="original_mesh")
        self.scene_widget2 = trimesh.viewer.SceneWidget(scene)
        hbox.add(self.scene_widget2)

        gui.add(hbox)

        pyglet.app.run()

    def cursor_triangle(self):
        x, y = self.mouse_pos
        origins, vectors, _ = self.scene_widget1.scene.camera_rays()
        resolution = self.scene_widget1.scene.camera.resolution
        x_in_scene, y_in_scene = x - self.padding, y - self.padding
        idx = x_in_scene * resolution[1] + ((resolution[1] - 1) - y_in_scene)
        current_origin, current_vector = origins[idx], vectors[idx]
        # try:
        _, _, triangle_index = self.original_mesh.ray.intersects_location(
            [current_origin], [current_vector], multiple_hits=False
        )
        return triangle_index[0]

    def recenter(self):
        previous_centroid = self.scene_widget1.scene.centroid
        triangle_index = self.cursor_triangle()
        if "bounding_box_needed_for_centering" not in self.scene_widget1.scene.geometry:
            geom = trimesh.path.creation.box_outline((1000, 1000, 1000))
            self.scene_widget1.scene.add_geometry(
                geom, geom_name="bounding_box_needed_for_centering"
            )
            self.previous_offset = 0, 0, 0
        geom = self.scene_widget1.scene.geometry["bounding_box_needed_for_centering"]
        geom.vertices -= self.previous_offset

        center = self.original_mesh.triangles_center[triangle_index]
        geom.vertices += center
        self.previous_offset = center

        self.scene_widget1.scene.camera_transform[:3, 3] += center - previous_centroid
        self.scene_widget1._initial_camera_transform = (
            self.scene_widget1.scene.camera_transform
        )

        self.scene_widget1.reset_view()

    def fit(self):
        # self.classifier = svm.SVC()
        self.classifier = ensemble.RandomForestClassifier()

        vertex_indices = list(self.vertex_indices_to_group_dict.keys())
        groups = list(self.vertex_indices_to_group_dict.values())
        data = list(
            zip(
                self.mean_curvature[vertex_indices],
                self.gaussian_curvature[vertex_indices],
                self.thickness[vertex_indices],
                self.obscurance[vertex_indices],
            )
        )
        self.scaler = preprocessing.StandardScaler().fit(data)
        data_transformed = self.scaler.transform(data)
        self.classifier.fit(data_transformed, groups)

    def predict(self):
        test = list(
            zip(
                self.mean_curvature,
                self.gaussian_curvature,
                self.thickness,
                self.obscurance,
            )
        )
        test_transformed = self.scaler.transform(test)
        group_predictions = self.classifier.predict(test_transformed)
        colors = [self.group_colors[group] for group in group_predictions]

        vertex_colors = [(0.5, 0.5, 0.5, 1.0)] * len(
            self.original_mesh_window2.vertices
        )

        self.original_mesh_window2.visual.vertex_colors = colors  # vertex_colors
        self.scene_widget2._draw()

    def _create_window(self, width, height):
        try:
            config = pyglet.gl.Config(
                sample_buffers=1, samples=4, depth_size=24, double_buffer=True
            )
            window = pyglet.window.Window(config=config, width=width, height=height)
        except pyglet.window.NoSuchConfigException:
            config = pyglet.gl.Config(double_buffer=True)
            window = pyglet.window.Window(config=config, width=width, height=height)

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
                    self.fit()
                    print("fitted")
                    self.predict()
                    print("predicted")

        @window.event
        def on_mouse_motion(x, y, dx, dy):
            self.mouse_pos = x, y
            if self.key_pressed:
                triangle_index = self.cursor_triangle()

                redraw = False

                geom_names_to_delete = []
                current_group = self.key_pressed
                geom_name = f"{current_group}_{triangle_index}"
                if triangle_index not in self.triangle_indices_to_group_dict:
                    redraw = True
                else:
                    redraw = False
                    previous_group = self.triangle_indices_to_group_dict[triangle_index]
                    if previous_group != current_group:
                        # remove previous one (have to translate otherwise it still shows up)
                        previous_geom_name = f"{previous_group}_{triangle_index}"
                        self.scene_widget1.scene.geometry[
                            previous_geom_name
                        ].apply_translation((-1000, -1000, -1000))
                        geom_names_to_delete.append(previous_geom_name)
                        redraw = True

                if redraw:
                    # self.scene_widget1.scene.delete_geometry("original_mesh")
                    self.triangle_indices_to_group_dict[triangle_index] = current_group
                    for v in self.original_mesh.faces[triangle_index]:
                        self.vertex_indices_to_group_dict[v] = current_group

                    submesh = self.original_mesh_scaled.submesh([[triangle_index]])[0]
                    submesh.visual.face_colors = self.group_colors[current_group]
                    self.scene_widget1.scene.add_geometry(submesh, geom_name=geom_name)
                    self.scene_widget1._draw()
                    self.scene_widget1.scene.delete_geometry(geom_names_to_delete)

        return window


if __name__ == "__main__":
    np.random.seed(0)
    Application()

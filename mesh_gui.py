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
from sklearn import svm, preprocessing

import numba
from numba import jit
from numba.typed import Dict
import pymeshlab

here = pathlib.Path(__file__).resolve().parent


def average_over_one_ring(mesh, c):
    g = nx.from_edgelist(mesh.edges_unique)
    one_rings = [list(g[i].keys()) for i in range(len(mesh.vertices))]

    avg = [0] * len(c)
    for vertex_id, one_ring in enumerate(one_rings):
        avg[vertex_id] = np.mean(c[one_ring])

    return np.array(avg)


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


# @jit(nopython=True)
# def jit_my_discrete_mean_curvature_measure(
#     g,
#     face_angles_row,
#     face_angles_col,
#     face_angles_data,
#     fa,
#     fae,
#     fau,
#     vertices,
#     vertex_faces,
#     area_faces,
# ):
#     """Calculate discrete mean curvature of mesh using one-ring neighborhood."""

#     # one-rings (immediate neighbors of) each vertex
#     # g = nx.from_edgelist(edges_unique)

#     # cotangents of angles and store in dictionary based on corresponding vertex and face
#     keys = tuple(zip(face_angles_row, face_angles_col))
#     cotangents = dict(zip(keys, 1 / np.tan(face_angles_data)))

#     # discrete Laplace-Beltrami contribution of the shared edge of adjacent faces:
#     #        /*\
#     #       / * \
#     #      /  *  \
#     #    vi___*___vj
#     #
#     # store results in dictionary with vertex ids as keys
#     keys = tuple(zip(fae[:, 0], fae[:, 1]))
#     cotangent_sums = np.array(
#         [
#             cotangents[(v[0], fa[i][0])] + cotangents[(v[1], fa[i][1])]
#             for i, v in enumerate(fau)
#         ]
#     )
#     edge_measure = dict(
#         zip(
#             keys,
#             (vertices[fae[:, 1]] - vertices[fae[:, 0]]) * cotangent_sums[:, None],
#         )
#     )

#     # calculate mean curvature using one-ring
#     mean_curv = [0] * len(vertices)
#     for vertex_id, face_ids in enumerate(vertex_faces):
#         face_ids = face_ids[face_ids != -1]  # faces associated with vertex_id
#         one_ring = list(g[vertex_id].keys())
#         delta_s = 0

#         for one_ring_vertex_id in one_ring:
#             if (vertex_id, one_ring_vertex_id) in edge_measure:
#                 delta_s += edge_measure[(vertex_id, one_ring_vertex_id)]
#             elif (one_ring_vertex_id, vertex_id) in edge_measure:
#                 delta_s -= edge_measure[(one_ring_vertex_id, vertex_id)]

#         delta_s *= 1 / (2 * np.sum(area_faces[face_ids]) / 3)  # use 1/3 of the areas
#         mean_curv[vertex_id] = 0.5 * np.linalg.norm(delta_s)

#     return np.array(mean_curv)


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


def create_scene():
    """
    Create a scene with a Fuze bottle, some cubes, and an axis.
    Returns
    ----------
    scene : trimesh.Scene
      Object with geometry
    """
    scene = trimesh.Scene()

    # geom = trimesh.load(str(here / "content/MPI-FAUST/meshes/tr_reg_000.ply"))
    geom = trimesh.load(str("/Users/ackermand/Documents/Downloads/410_roi1.obj"))
    geom.vertices = (geom.vertices - np.min(geom.vertices, axis=0)) / (
        np.amax(geom.vertices) - np.min(geom.vertices, axis=0)
    )
    print(np.max(geom.vertices, axis=0), np.min(geom.vertices, axis=0))
    geom.visual.vertex_colors = [0.5, 0.5, 0.5, 1.0]
    scene.add_geometry(geom, geom_name="original_mesh")

    return scene


class Application:

    """
    Example application that includes moving camera, scene and image update.
    """

    def __init__(self):
        # geom = trimesh.load(str(here / "content/MPI-FAUST/meshes/tr_reg_000.ply"))
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh("./content/MPI-FAUST/meshes/tr_reg_000.ply")
        mesh = ms.current_mesh()
        t = time.time()
        ms.meshing_repair_non_manifold_edges()
        ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=0)
        # jit_my_discrete_mean_curvature_measure(
        #     list(nx.to_dict_of_lists(nx.from_edgelist(m.edges_unique)).values()),
        #     m.face_angles_sparse.row,
        #     m.face_angles_sparse.col,
        #     m.face_angles_sparse.data,
        #     m.face_adjacency,
        #     m.face_adjacency_edges,
        #     m.face_adjacency_unshared,
        #     m.vertices,
        #     m.vertex_faces,
        #     m.area_faces,
        # )
        mean_curvature = mesh.vertex_scalar_array()
        # old = old_my_discrete_mean_curvature_measure(self.original_mesh)
        # print("equal?", np.array_equal(mean_curvature, old))
        # mean_curvature[0]=np.NaN
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
        geom.visual.vertex_colors = [0.5, 0.5, 0.5, 1.0]
        scene.add_geometry(geom, geom_name="original_mesh")

        self.original_mesh = scene.geometry["original_mesh"]

        self.mean_curvature = average_over_one_ring(self.original_mesh, mean_curvature)
        print("mca", time.time() - t)
        t = time.time()
        self.gaussian_curvature = average_over_one_ring(
            self.original_mesh, gaussian_curvature
        )
        print("gca", time.time() - t)
        t = time.time()
        self.thickness = average_over_one_ring(self.original_mesh, thickness)
        print("tha", time.time() - t)
        t = time.time()

        invalids = (
            np.isnan(self.mean_curvature)
            | np.isinf(self.mean_curvature)
            | np.isnan(self.gaussian_curvature)
            | np.isinf(self.gaussian_curvature)
            | np.isnan(self.thickness)
            | np.isinf(self.thickness)
        )
        self.valid_vertices = {
            i for i, is_invalid in enumerate(invalids) if ~is_invalid
        }
        print(len(self.original_mesh.vertices), len(self.valid_vertices))
        self.original_mesh_scaled = self.original_mesh.copy()
        self.original_mesh_scaled.vertices += (
            self.original_mesh_scaled.vertex_normals * 0.0001
        )
        self.triangle_group_assignments = [] * len(self.original_mesh_scaled.faces)
        self.scene_widget1 = trimesh.viewer.SceneWidget(scene)
        self.scene_widget1.scene.camera._fov = [45.0, 45.0]
        # print(self.scene_widget1.scene.camera.resolution)
        # self.scene_widget1.scene.camera.fov = 60 * (self.scene_widget1.scene.camera.resolution /
        #                      self.scene_widget1.scene.camera.resolution.max())
        # self.scene_widget1._angles = [np.deg2rad(45), 0, 0]
        hbox.add(self.scene_widget1)

        # scene widget for changing scene
        scene = trimesh.Scene()
        # geom = trimesh.path.creation.box_outline((0.6, 0.6, 0.6))
        # scene.add_geometry(geom)
        self.original_mesh_window2 = self.original_mesh.copy()
        scene.add_geometry(self.original_mesh_window2, geom_name="original_mesh")
        self.scene_widget2 = trimesh.viewer.SceneWidget(scene)
        hbox.add(self.scene_widget2)

        # integrate with other widget than SceneWidget
        # self.image_widget = glooey.Image()
        # hbox.add(self.image_widget)

        gui.add(hbox)

        ## pyglet.clock.schedule_interval(self.callback, 1. / 20)

        pyglet.app.run()

    def fit(self):
        self.svc = svm.SVC()

        vertex_indices = list(self.vertex_indices_to_group_dict.keys())
        groups = list(self.vertex_indices_to_group_dict.values())
        data = list(
            zip(
                self.mean_curvature[vertex_indices],
                self.gaussian_curvature[vertex_indices],
                self.thickness[vertex_indices],
            )
        )
        self.scaler = preprocessing.StandardScaler().fit(data)
        data_transformed = self.scaler.transform(data)
        self.svc.fit(data_transformed, groups)

    def predict(self):
        test = list(
            zip(
                self.mean_curvature[list(self.valid_vertices)],
                self.gaussian_curvature[list(self.valid_vertices)],
                self.thickness[list(self.valid_vertices)],
            )
        )
        test_transformed = self.scaler.transform(test)
        group_predictions = self.svc.predict(test_transformed)
        colors = [self.group_colors[group] for group in group_predictions]

        # self.scene_widget2.scene.geometry["original_mesh"].apply_translation((-1000,-1000,-1000))
        # self.scene_widget2.scene.delete_geometry("original_mesh")
        vertex_colors = [(0.5, 0.5, 0.5, 1.0)] * len(
            self.original_mesh_window2.vertices
        )
        for idx, valid_vertex in enumerate(self.valid_vertices):
            vertex_colors[valid_vertex] = colors[idx]

        self.original_mesh_window2.visual.vertex_colors = vertex_colors
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
                    print(
                        len(self.scene_widget1.scene.camera_rays()[0]),
                        self.scene_widget1.scene.camera.resolution,
                    )
                if symbol == pyglet.window.key.P:
                    self.fit()
                    print("fitted")
                    self.predict()
                    print("predicted")

        @window.event
        def on_mouse_motion(x, y, dx, dy):
            if self.key_pressed:
                origins, vectors, _ = self.scene_widget1.scene.camera_rays()
                resolution = self.scene_widget1.scene.camera.resolution
                x_in_scene, y_in_scene = x - self.padding, y - self.padding
                idx = x_in_scene * resolution[1] + ((resolution[1] - 1) - y_in_scene)
                current_origin, current_vector = origins[idx], vectors[idx]
                # try:
                _, _, triangle_index = self.original_mesh.ray.intersects_location(
                    [current_origin], [current_vector], multiple_hits=False
                )
                triangle_index = triangle_index[0]
                # use_sphere = True
                # if use_sphere:
                #     if "cursor_sphere" in self.scene_widget1.scene.geometry:
                #         cursor_sphere = self.scene_widget1.scene.geometry["cursor_sphere"]
                #     else:
                #         cursor_sphere = trimesh.creation.icosphere(radius=0.001)
                #         cursor_sphere.visual.face_colors = (1.0,0,0,0.2)
                #         self.scene_widget1.scene.add_geometry(cursor_sphere,geom_name="cursor_sphere")
                #     cursor_sphere.apply_translation(-cursor_sphere.center_mass+original_mesh.triangles_center[triangle_index])
                # else:
                # self.scene_widget1.scene.delete_geometry("submesh")
                redraw = False
                # print("before",self.triangle_indices_by_group)

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
                        if v in self.valid_vertices:
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

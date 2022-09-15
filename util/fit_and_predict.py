from functools import partial
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import copy


class FitAndPredict:
    def __init__(self, df, np):
        self.df = df
        self.viewer = np.viewer
        self.mesh_id_to_index_dict = np.mesh_id_to_index_dict

    def fit(self, mesh_index_to_class_dict):
        self.classifier = MLPClassifier(alpha=1, max_iter=1000)
        classes = list(mesh_index_to_class_dict.values())

        mesh_indices = list(mesh_index_to_class_dict.keys())
        data = self.metrics[mesh_indices, :]
        self.scaler = preprocessing.StandardScaler().fit(data)
        data_transformed = self.scaler.transform(data)
        self.classifier.fit(data_transformed, classes)

    def predict(self):
        test_transformed = self.scaler.transform(self.metrics)
        class_predictions = self.classifier.predict(test_transformed)
        return class_predictions

    def set_metrics(self, metric_names):
        def fit_and_predict(s):
            self.metrics = self.df[metric_names].to_numpy()
            class_layers = ["class h", "class j", "class k", "class l"]
            mesh_index_to_class_dict = {}
            state = self.viewer.state
            for layer in state.layers:
                if layer.name != "mesh":  # then is a class layer
                    for segment_id in layer.segments:
                        class_id = class_layers.index(layer.name)
                        mesh_index = self.mesh_id_to_index_dict[segment_id]
                        mesh_index_to_class_dict[mesh_index] = class_id
            self.fit(mesh_index_to_class_dict)
            class_predictions = self.predict()

            new_state = copy.deepcopy(self.viewer.state)
            mesh_layer = new_state.layers["mesh"]

            colors = ["red", "gray", "blue", "magenta"]

            segment_colors = {}
            for segment_id in mesh_layer.segments:
                mesh_index = self.mesh_id_to_index_dict[int(segment_id)]
                class_id = class_predictions[mesh_index]
                segment_colors[segment_id] = colors[class_id]

            mesh_layer.segment_colors = segment_colors
            self.viewer.set_state(new_state)

        # self.viewer.actions.remove("my-action-p", fit_and_predict)
        self.viewer.actions.add("my-action-p", fit_and_predict)

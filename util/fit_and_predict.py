from functools import partial
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import copy
from datetime import datetime

from util.neuroglancer_predictor import NeuroglancerPredictor
import pandas as pd
import os

class FitAndPredict:
    def __init__(self, df: pd.DataFrame, np: NeuroglancerPredictor):
        self.df = df
        self.class_info = np.class_info
        self.class_names = [class_info[0] for class_info in np.class_info]
        self.class_colors = [class_info[2] for class_info in np.class_info]
        self.viewer = np.viewer
        self.segment_ids = np.segment_ids
        self.mesh_id_to_index_dict = np.mesh_id_to_index_dict

        self.output_dir = f"output/classification/{np.cell}/{np.organelle}/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)

    def fit(self):
        self.classifier = MLPClassifier(alpha=1, max_iter=1000)
        classes = list(self.mesh_index_to_class_dict.values())
        mesh_indices = list(self.mesh_index_to_class_dict.keys())
        data = self.metrics[mesh_indices, :]
        self.scaler = preprocessing.StandardScaler().fit(data)
        data_transformed = self.scaler.transform(data)
        self.classifier.fit(data_transformed, classes)

    def predict(self):
        test_transformed = self.scaler.transform(self.metrics)
        self.class_predictions = self.classifier.predict(test_transformed)
    
    def write_output(self):
        manually_labeled_class_names=["None"]*len(self.segment_ids)
        class_prediction_names = [self.class_names[class_prediction] for class_prediction in self.class_predictions]
        for manual_labeled_mesh_index,manual_labeled_class_index in self.mesh_index_to_class_dict.items():
            manually_labeled_class_names[manual_labeled_mesh_index] = self.class_names[manual_labeled_class_index]
            # ensure manually labeled classes are kept
            class_prediction_names[manual_labeled_mesh_index] = self.class_names[manual_labeled_class_index]
            self.class_predictions[manual_labeled_mesh_index] = manual_labeled_class_index


        classificaiton_df = pd.DataFrame({"Object ID": self.segment_ids,
                                          "Manually Labeled Class": manually_labeled_class_names,
                                          "Class Prediction": self.class_predictions,
                                          "Class Name": class_prediction_names})
        
        classificaiton_df.to_csv(f"{self.output_dir}/classification.csv",index=False)

    def set_metrics(self, metric_names):
        def fit_and_predict(s):
            self.metrics = self.df[metric_names].to_numpy()            
            self.mesh_index_to_class_dict = {}
            state = self.viewer.state
            for layer in state.layers:
                if layer.name != "mesh":  # then is a class layer
                    for segment_id in layer.segments:
                        class_id = self.class_names.index(layer.name)
                        mesh_index = self.mesh_id_to_index_dict[segment_id]
                        self.mesh_index_to_class_dict[mesh_index] = class_id
            self.fit()
            self.predict()
            self.write_output()



            new_state = copy.deepcopy(self.viewer.state)
            mesh_layer = new_state.layers["mesh"]

            segment_colors = {}
            for segment_id in mesh_layer.segments:
                mesh_index = self.mesh_id_to_index_dict[int(segment_id)]
                class_id = self.class_predictions[mesh_index]
                segment_colors[segment_id] = self.class_colors[class_id]

            mesh_layer.segment_colors = segment_colors
            self.viewer.set_state(new_state)

        # self.viewer.actions.remove("my-action-p", fit_and_predict)
        self.viewer.actions.add("my-action-p", fit_and_predict)

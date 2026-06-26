"""Interactive mesh classification driver.

Reads dataset paths from a gitignored config.yaml (see config.example.yaml), so
no absolute data paths or dataset names live in the committed code:

    pixi run python classify.py <dataset_key>

Then in the viewer: hover a mesh and press a class hotkey to label it, `u` to
unclassify, and `p` to fit on the labeled meshes and predict the rest.
"""
import sys

import pandas as pd

import util.neuroglancer_predictor as NP
import util.fit_and_predict as FP
from util.config import load_dataset_config


def main():
    if len(sys.argv) < 2:
        sys.exit(
            "usage: python classify.py <dataset_key>  (keys are defined in config.yaml)"
        )
    cfg = load_dataset_config(sys.argv[1])

    df = pd.read_csv(cfg["metrics_csv"])
    class_info = [tuple(c) for c in cfg["classes"]]

    predictor = NP.NeuroglancerPredictor(
        cfg["dataset"],
        cfg["organelle"],
        class_info=class_info,
        segmentation_path=cfg["segmentation_path"],
        mesh_path=cfg["mesh_path"],
    )
    predictor.setup_neuroglancer()
    fp = FP.FitAndPredict(df, predictor)
    fp.set_metrics(list(df.columns[1:]))

    print("viewer:", predictor.viewer.get_viewer_url())
    input("Press Enter to quit and stop the viewer...")


if __name__ == "__main__":
    main()

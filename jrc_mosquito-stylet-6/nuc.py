# %%
from importlib import reload
import util.neuroglancer_predictor as NP

reload(NP)
import util.fit_and_predict as FP

reload(FP)
import pandas as pd

dataset = "jrc_mosquito-stylet-6"
organelle = "nuc"

# mesh metrics produced by mesh-n-bone (sharded multiresolution meshes)
df_mesh = pd.read_csv(
    f"/nrs/cellmap/ackermand/new_meshes/mesh-n-bone/{dataset}/{organelle}/metrics/mesh_metrics.csv"
)
df = df_mesh
# %%

np = NP.NeuroglancerPredictor(
    dataset,
    organelle,
    class_info=[
        ("neuron (h, red)", "h", "red"),
        ("non_neuron (j, gray)", "j", "gray"),
    ],
    segmentation_path=f"/nrs/cellmap/ackermand/symlinks/{dataset}/{dataset}.zarr/recon-1/labels/inference/segmentations/{organelle}",
    mesh_path=f"/nrs/cellmap/ackermand/new_meshes/mesh-n-bone/{dataset}/{organelle}/multires",
)
np.setup_neuroglancer()
print("fit and predict")
fp = FP.FitAndPredict(df, np)
fp.set_metrics(list(df.columns[1:]))
print("set metrics")

# %%
if __name__ == "__main__":
    # When run as a plain script (`pixi run python jrc_mosquito-stylet-6/nuc.py`),
    # keep the process alive so the neuroglancer viewer server stays up. Not
    # needed when running the cells interactively (the kernel stays alive).
    print("viewer:", np.viewer.get_viewer_url())
    input("Press Enter to quit and stop the viewer...")

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
        ("good big (h, red)", "h", "red"),
        ("bad big (j, gray)", "j", "gray"),
        ("good small (k, blue)", "k", "blue"),
        ("bad small (l, magenta)", "l", "magenta"),
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

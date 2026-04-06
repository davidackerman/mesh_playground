# %%
from importlib import reload
import util.neuroglancer_predictor as NP

reload(NP)
import util.fit_and_predict as FP

reload(FP)
import pandas as pd

dataset = "jrc_mus-salivary-1"
organelle = "nuc"

df_mesh = pd.read_csv(
    f"/nrs/cellmap/zubovy/temp/single_res_meshes/{dataset}/inference/segmentations/nuc/ply/metrics/mesh_metrics.csv"
)
# df_skeleton = pd.read_csv(
#     "/nrs/cellmap/ackermand/new_meshes/skeletons/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/metrics/skeleton_metrics.csv"
# )
# combine dataframes based on id column
# df = pd.merge(df_mesh, on="id")
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
    segmentation_path=f"/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/labels/inference/segmentations/{organelle}",
    mesh_path=f"/nrs/cellmap/data/{dataset}/neuroglancer/mesh/inference/segmentations/{organelle}",
)
np.setup_neuroglancer()
print("fit and predict")
fp = FP.FitAndPredict(df, np)
fp.set_metrics(list(df.columns[1:]))
print("set metrics")

# %%

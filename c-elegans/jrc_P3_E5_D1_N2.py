# %%
from importlib import reload
import util.neuroglancer_predictor as NP

reload(NP)
import util.fit_and_predict as FP

reload(FP)
import pandas as pd

dataset = "c-elegans/jrc_P3_E5_D1_N2"
organelle = "nuc_filled_ply"

df_mesh = pd.read_csv(
    f"/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/c-elegans/jrc_P3_E5_D1_N2/nuc_filled_ply/metrics/mesh_metrics.csv"
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
    segmentation_path="/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_P3_E5_D1_N2/jrc_P3_E5_D1_N2.zarr/nuc_filled",
)
np.setup_neuroglancer()
print("fit and predict")
fp = FP.FitAndPredict(df, np)
fp.set_metrics(list(df.columns[1:]))
print("set metrics")

# %%
import pandas as pd

df = pd.read_csv(
    "/groups/scicompsoft/home/ackermand/Programming/mesh_playground/output/classification/c-elegans/jrc_P3_E5_D1_N2/nuc_filled_ply/20250812_160025/classification.csv"
)
# only keep those that have column "Manually Labeled Class" contains the word good
df = df[df["Manually Labeled Class"].str.contains("good")]
df.to_csv(
    "/groups/scicompsoft/home/ackermand/Programming/mesh_playground/output/classification/c-elegans/jrc_P3_E5_D1_N2/nuc_filled_ply/20250812_160025/ids_to_keep.csv", index=False
)
# %%

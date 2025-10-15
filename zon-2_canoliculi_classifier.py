#  %% NOTE: Original, first 1000 stuff
# # %%
# from importlib import reload
# import util.neuroglancer_predictor as NP

# reload(NP)
# import util.fit_and_predict as FP

# reload(FP)
# import pandas as pd

# dataset = "jrc_mus-liver-zon-2"
# organelle = "/canoliculi_cc_close_raw_mask_filled"

# df_mesh = pd.read_csv(
#     "/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/metrics/mesh_metrics.csv"
# )
# df_skeleton = pd.read_csv(
#     "/nrs/cellmap/ackermand/new_meshes/skeletons/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/metrics/skeleton_metrics.csv"
# )
# # combine dataframes based on id column
# df = pd.merge(df_mesh, df_skeleton, on="id")
# # get top 1000 ids based on volume (nm^3) column
# df_largest = df.nlargest(1000, "volume (nm^3)")
# # get ids
# largest_ids = df_largest["id"].tolist()
# # %%

# np = NP.NeuroglancerPredictor(
#     dataset,
#     organelle,
#     class_info=[
#         ("good big (h, red)", "h", "red"),
#         ("bad big (j, gray)", "j", "gray"),
#         ("good small (k, blue)", "k", "blue"),
#         ("bad small (l, magenta)", "l", "magenta"),
#     ],
#     selected_segment_ids=largest_ids,
# )
# np.setup_neuroglancer()
# print("fit and predict")
# fp = FP.FitAndPredict(df, np)
# fp.set_metrics(list(df.columns[1:]))
# print("set metrics")

# %% NOTE: To second largest 1000 using first 1000 as ground truth
# from importlib import reload
# import util.neuroglancer_predictor as NP

# reload(NP)
# import util.fit_and_predict as FP

# reload(FP)
# import pandas as pd

# dataset = "jrc_mus-liver-zon-2"
# organelle = "/canoliculi_cc_close_raw_mask_filled"

# df_mesh = pd.read_csv(
#     "/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/metrics/mesh_metrics.csv"
# )
# df_skeleton = pd.read_csv(
#     "/nrs/cellmap/ackermand/new_meshes/skeletons/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/metrics/skeleton_metrics.csv"
# )
# # combine dataframes based on id column
# df = pd.merge(df_mesh, df_skeleton, on="id")
# # get second largest 1000 ids based on volume (nm^3) column
# df_sorted = df.sort_values(by="volume (nm^3)", ascending=False)
# df_second_largest_thousand = df_sorted.iloc[1000:2000]
# # get ids
# largest_ids = df_second_largest_thousand["id"].tolist()
# # %%
# # combine the two dataframes based on

# np = NP.NeuroglancerPredictor(
#     dataset,
#     organelle,
#     class_info=[
#         ("good big (h, red)", "h", "red"),
#         ("bad big (j, gray)", "j", "gray"),
#         ("good small (k, blue)", "k", "blue"),
#         ("bad small (l, magenta)", "l", "magenta"),
#     ],
#     selected_segment_ids=largest_ids,
#     previous_results="/groups/scicompsoft/home/ackermand/Programming/mesh_playground/output/classification/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/20250128_124825/classification.csv",
# )
# np.setup_neuroglancer()
# print("fit and predict")
# fp = FP.FitAndPredict(df, np)
# fp.set_metrics(list(df.columns[1:]))
# print("set metrics")

# %%
# %% NOTE: Now do it with top 1000 good
# from importlib import reload
# import util.neuroglancer_predictor as NP

# reload(NP)
# import util.fit_and_predict as FP

# reload(FP)
# import pandas as pd

# dataset = "jrc_mus-liver-zon-2"
# organelle = "/canoliculi_cc_close_raw_mask_filled"

# df_mesh = pd.read_csv(
#     "/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/metrics/mesh_metrics.csv"
# )
# df_skeleton = pd.read_csv(
#     "/nrs/cellmap/ackermand/new_meshes/skeletons/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/metrics/skeleton_metrics.csv"
# )
# # combine dataframes based on id column
# df = pd.merge(df_mesh, df_skeleton, on="id")
# second_round_categorizations_df = pd.read_csv(
#     "/groups/scicompsoft/home/ackermand/Programming/mesh_playground/output/classification/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/20250129_112159/classification.csv"
# )
# # count the number of rows that have "good" in the Class Name column
# df_good = df[second_round_categorizations_df["Class Name"].str.contains("good")]

# # get top 1000 that have "good" in the Class Name column
# df_good_largest = df_good.nlargest(1000, "volume (nm^3)")
# np = NP.NeuroglancerPredictor(
#     dataset,
#     organelle,
#     class_info=[
#         ("good big (h, red)", "h", "red"),
#         ("bad big (j, gray)", "j", "gray"),
#         ("good small (k, blue)", "k", "blue"),
#         ("bad small (l, magenta)", "l", "magenta"),
#     ],
#     selected_segment_ids=df_good_largest["id"].tolist(),
#     previous_results="/groups/scicompsoft/home/ackermand/Programming/mesh_playground/output/classification/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/20250128_124825/classification.csv",
# )
# np.setup_neuroglancer()
# print("fit and predict")
# fp = FP.FitAndPredict(df, np)
# fp.set_metrics(list(df.columns[1:]))
# print("set metrics")
# %%
from importlib import reload
import util.neuroglancer_predictor as NP

reload(NP)
import util.fit_and_predict as FP

reload(FP)
import pandas as pd

dataset = "jrc_mus-liver-zon-2"
organelle = "/canoliculi_cc_close_raw_mask_filled"

df_mesh = pd.read_csv(
    "/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/metrics/mesh_metrics.csv"
)
df_skeleton = pd.read_csv(
    "/nrs/cellmap/ackermand/new_meshes/skeletons/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/metrics/skeleton_metrics.csv"
)
# combine dataframes based on id column
df = pd.merge(df_mesh, df_skeleton, on="id")
thrid_round_categorizations_df = pd.read_csv(
    "/groups/scicompsoft/home/ackermand/Programming/mesh_playground/output/classification/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/20250129_125415/classification.csv"
)
# count the number of rows that have "good" in the Class Name column
df_good = df[thrid_round_categorizations_df["Class Name"].str.contains("good")]

# get top 1000 that have "good" in the Class Name column
df_good_largest = df_good.nlargest(1000, "volume (nm^3)")
np = NP.NeuroglancerPredictor(
    dataset,
    organelle,
    class_info=[
        ("good big (h, red)", "h", "red"),
        ("bad big (j, gray)", "j", "gray"),
        ("good small (k, blue)", "k", "blue"),
        ("bad small (l, magenta)", "l", "magenta"),
    ],
    selected_segment_ids=df_good_largest["id"].tolist(),
    previous_results="/groups/scicompsoft/home/ackermand/Programming/mesh_playground/output/classification/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/20250128_124825/classification.csv",
)
np.setup_neuroglancer()
print("fit and predict")
fp = FP.FitAndPredict(df, np)
fp.set_metrics(list(df.columns[1:]))
print("set metrics")

# %%
# Save final version as all the good ones
import pandas as pd

df = pd.read_csv(
    "/groups/scicompsoft/home/ackermand/Programming/mesh_playground/output/classification/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/20250129_125415/classification.csv",
)
# filter by Class Name column having "good" in it
df_good = df[df["Class Name"].str.contains("good")]
df_good.to_csv(
    "/groups/scicompsoft/home/ackermand/Programming/mesh_playground/output/classification/jrc_mus-liver-zon-2/canoliculi_cc_close_raw_mask_filled/20250129_125415/classification_good.csv",
    index=False,
)



# %%

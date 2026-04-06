# mesh_playground

Interactive classification and analysis of 3D mesh objects from electron microscopy segmentations. Designed for morphological classification of cellular organelles (canaliculi, nuclei, mitochondria) using a Neuroglancer-based annotation workflow and machine learning.

## Features

- **Mesh metric computation**: volume, surface area, curvature (mean, Gaussian, RMS), shape diameter function (thickness), principal inertia, and skeletal/morphological features
- **Interactive annotation**: web-based Neuroglancer viewer with keyboard-driven classification into user-defined classes
- **ML classification**: trains an MLP classifier on manually labeled mesh features to predict classes for unlabeled meshes
- **Iterative refinement**: reload previous classification results as ground truth and refine with additional rounds of annotation
- **Parallel processing**: Dask-based delayed evaluation for batch mesh metric computation

## Datasets

Example scripts are included for:

- Mouse liver canaliculi (`jrc_mus-liver-zon-1`, `jrc_mus-liver-zon-2`)
- Mouse salivary gland nuclei (`jrc_mus-salivary-1`, `jrc_mus-salivary-2`, `jrc_mus-salivary-3`)
- C. elegans nuclei (`jrc_P3_E5_D1_N2`)

## Project Structure

```
mesh_playground/
├── util/
│   ├── mesh.py                    # Mesh loading, repair, and metric computation
│   ├── neuroglancer_predictor.py  # Neuroglancer viewer setup and annotation
│   └── fit_and_predict.py         # ML training and prediction pipeline
├── environment.yaml               # Conda environment specification
└── output/                        # Classification results (CSV)
```

## Examples

Dataset-specific classification scripts demonstrate the full workflow for each organism and organelle:

| Script | Dataset | Organelle | Description |
|--------|---------|-----------|-------------|
| `jrc_mus-salivary-1/nuc.py` | Mouse salivary gland 1 | Nuclei | Single-round nuclei classification |
| `jrc_mus-salivary-2/nuc.py` | Mouse salivary gland 2 | Nuclei | Single-round nuclei classification |
| `jrc_mus-salivary-3/nuc.py` | Mouse salivary gland 3 | Nuclei | Single-round nuclei classification |
| `c-elegans/jrc_P3_E5_D1_N2.py` | C. elegans | Nuclei | Nuclei classification with post-hoc filtering of "good" results |
| `zon-1_canaliculi_classifier.py` | Mouse liver zone 1 | Canaliculi | Canaliculi classification with custom segmentation path |
| `zon-2_canoliculi_classifier.py` | Mouse liver zone 2 | Canaliculi | Multi-round iterative refinement (see also `show_yurii.py`) |
| `show_yurii.py` | Mouse liver zone 2 | Canaliculi | Full multi-round workflow: classify top 1000 by volume, refine next 1000 using prior results, then re-classify top "good" predictions |

## Setup

Create the conda environment:

```bash
conda env create -f environment.yaml
```

Or without pinned versions:

```bash
conda env create -f environment_no_versions.yaml
```

## Usage

### 1. Compute mesh metrics

```python
from util.mesh import Mesh

mesh = Mesh(mesh_path, compute_skeleton=False)
metrics = mesh.get_metrics()
```

### 2. Interactive annotation

```python
from util.neuroglancer_predictor import NeuroglancerPredictor

np = NeuroglancerPredictor(
    dataset="jrc_mus-liver-zon-1",
    organelle="canaliculi",
    class_info=[
        ("good big (h, red)", "h", "red"),
        ("bad big (j, gray)", "j", "gray"),
        ("good small (k, blue)", "k", "blue"),
        ("bad small (l, magenta)", "l", "magenta"),
    ]
)
np.setup_neuroglancer()
```

Open the printed Neuroglancer URL in a browser. All meshes are displayed in the "all meshes" layer. Select a mesh and press one of the classification keys to move it into the corresponding class layer:

| Key | Action |
|-----|--------|
| `h` | Classify selected mesh as "good big" (red) |
| `j` | Classify selected mesh as "bad big" (gray) |
| `k` | Classify selected mesh as "good small" (blue) |
| `l` | Classify selected mesh as "bad small" (magenta) |
| `Shift+<key>` | Toggle visibility of the corresponding class layer |
| `p` | Fit classifier on manual labels and predict all remaining meshes |

The classification keys are defined by the `class_info` parameter — each tuple is `(class_name, key, color)`. When you press a key, the selected mesh is removed from its current layer and added to the target class layer with the specified color.

### 3. Train classifier and predict

After manually labeling a representative subset, press `p` to train an MLP classifier (scikit-learn `MLPClassifier`) on the labeled mesh metrics and predict classes for all unlabeled meshes. Predictions are color-coded in the "all meshes" layer and results are written to CSV.

```python
from util.fit_and_predict import FitAndPredict

fp = FitAndPredict(df_metrics, np)
fp.set_metrics(metric_columns)  # Binds the 'p' key to fit-and-predict
```

Results are exported to `output/classification/{dataset}/{organelle}/{timestamp}/classification.csv` with columns: Object ID, Manually Labeled Class, Class Prediction, and Class Name.

## Author

David Ackerman

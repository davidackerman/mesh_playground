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
├── jrc_mus-salivary-{1,2,3}/
│   └── nuc.py                     # Salivary gland nuclei classification scripts
├── c-elegans/
│   └── jrc_P3_E5_D1_N2.py        # C. elegans nuclei classification
├── zon-1_canaliculi_classifier.py # Liver zone-1 canaliculi classification
├── zon-2_canoliculi_classifier.py # Liver zone-2 canaliculi classification
├── show_yurii.py                  # Multi-round classification workflow example
├── mesh_gui.py                    # Dual-window mesh viewer with labeling
├── environment.yaml               # Conda environment specification
└── output/                        # Classification results (CSV)
```

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

Open the printed Neuroglancer URL in a browser and use the configured keyboard shortcuts to assign classes to meshes.

### 3. Train classifier and predict

```python
from util.fit_and_predict import FitAndPredict

fp = FitAndPredict(df_metrics, np)
fp.set_metrics(metric_columns)  # Trains on manual labels, predicts remaining
```

Results are exported to `output/classification/{dataset}/{organelle}/{timestamp}/classification.csv`.

## Author

David Ackerman

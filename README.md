# mesh_playground

Interactive classification and analysis of 3D mesh objects from electron microscopy segmentations. Designed for morphological classification of cellular organelles (e.g. nuclei, mitochondria) using a Neuroglancer-based annotation workflow and machine learning.

## Features

- **Mesh metric computation**: volume, surface area, curvature (mean, Gaussian, RMS), shape diameter function (thickness), principal inertia, and skeletal/morphological features
- **Interactive annotation**: web-based Neuroglancer viewer with keyboard-driven classification into user-defined classes
- **ML classification**: trains an MLP classifier on manually labeled mesh features to predict classes for unlabeled meshes
- **Iterative refinement**: reload previous classification results as ground truth and refine with additional rounds of annotation
- **Parallel processing**: Dask-based batch mesh metric computation

## Project Structure

```
mesh_playground/
├── analyze.py                     # Batch mesh metrics -> mesh_metrics.csv
├── classify.py                    # Annotation + classification driver (reads config.yaml)
├── config.example.yaml            # Template for config.yaml (gitignored)
├── util/
│   ├── mesh.py                    # Mesh loading, repair, and metric computation
│   ├── precomputed_mesh.py        # Neuroglancer precomputed mesh reader
│   ├── neuroglancer_predictor.py  # Neuroglancer viewer setup and annotation
│   ├── fit_and_predict.py         # ML training and prediction pipeline
│   └── config.py                  # config.yaml loader
└── output/                        # Metrics and classification results (CSV)
```

## Setup

Recommended: [pixi](https://pixi.sh) (reproducible from `pixi.lock`, Python 3.11 + NumPy 2):

```bash
pixi install          # build the environment
pixi run python ...   # run inside it (or `pixi shell`)
```

`pymeshlab` is installed from conda-forge (the PyPI wheel is not NumPy-2 compatible
and bundles an old libsqlite3 that breaks cloud-volume).

The thickness metric (pymeshlab's shape diameter function) needs OpenGL, so
`util/mesh.py` computes it under a virtual X display. Install the system `Xvfb`
binary (e.g. `apt install xvfb`) on headless machines.

Alternatively, the legacy conda environment:

```bash
conda env create -f environment.yaml
```

## Quick start on public data

`config.example.yaml` includes the mitochondria of a HeLa cell from OpenOrganelle (`jrc_hela-2`, 421 objects).

```bash
cp config.example.yaml config.yaml
pixi run python analyze.py \
    https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/neuroglancer/mesh/mito_seg \
    output/metrics/jrc_hela-2_mito
pixi run python classify.py jrc_hela-2_mito
```

## Usage

### 1. Compute mesh metrics

`analyze.py` reads every segment of a Neuroglancer precomputed mesh source (local path or URL) and writes `mesh_metrics.csv`, fanning the work out across processes with Dask:

```bash
pixi run python analyze.py <mesh_dir> <out_dir> [--lod 0] [--workers 10] [--skeletons]
```

### 2. Interactive annotation

Add an entry for your data to `config.yaml` (see `config.example.yaml`): the metrics CSV, the mesh source, optionally the EM volume and segmentation, and the classes as `[name, hotkey, color]`. Then:

```bash
pixi run python classify.py <dataset_key>
```

Open the printed Neuroglancer URL in a browser. All meshes are displayed in the "all meshes" layer. Hover over a mesh and press one of the classification keys to move it into the corresponding class layer:

| Key | Action |
|-----|--------|
| class hotkey (e.g. `h`, `j`) | Move the hovered mesh into that class's layer, in its color |
| `Shift+<key>` | Toggle visibility of the corresponding class layer |
| `u` | Move the hovered mesh back to the unlabeled pool |
| `p` | Fit classifier on manual labels and predict all remaining meshes |

If the data are on local disk, `path_rewrites` in `config.yaml` maps local path prefixes to the URL of a file server that can serve them to the viewer.

### 3. Train classifier and predict

After manually labeling a representative subset, press `p` to train an MLP classifier (scikit-learn `MLPClassifier`) on the labeled mesh metrics and predict classes for all unlabeled meshes. Predictions are color-coded in the "all meshes" layer, and each run is written to `output/classification/{dataset}/{organelle}/{session}/classification_{timestamp}.csv` with columns: Object ID, Manually Labeled Class, Class Prediction, and Class Name.

To continue from an earlier round, set `previous_results` in the dataset's config entry to one of those CSVs: its manual labels are loaded as ground truth for the next fit.

## Author

David Ackerman

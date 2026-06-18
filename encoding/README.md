# Encoding Pipeline

This directory contains the EEG encoding model implementation that was missing
from the public repository. It intentionally includes source code only. Datasets,
feature arrays, checkpoints, pretrained weights, logs, and generated results are
not committed.

## Directory Layout

- `synthesizing_eeg_data/`: linearizing/end-to-end EEG encoding training,
  evaluation, correlation analysis, and experiment shell entrypoints.
- `dnn_feature_maps_extraction/`: unified visual feature-map extraction and
  PCA before EEG encoding.
- `pretrain_weights/`: local-only weight download target. Ignored by Git.
- `DNNs/`: local-only feature/PCA/result output target. Ignored by Git.

## Data And Weight Inputs

By default, paths are resolved in `synthesizing_eeg_data/encoding_config.py`.
Override them with environment variables when running on a new machine:

```bash
export DATASET_ROOT=/path/to/datasets
export EEG_DATA_DIR=/path/to/THINGS_EEG/Preprocessed_data_250Hz
export IMAGE_SET_DIR=/path/to/THINGS_EEG/images_set
export PRETRAIN_WEIGHTS_DIR=/path/to/pretrain_weights
export PROJECT_DIR=/path/to/output/DNNs
```

Expected EEG files are per subject, for example:

```text
$EEG_DATA_DIR/sub-01/preprocessed_eeg_training.npy
$EEG_DATA_DIR/sub-01/preprocessed_eeg_test.npy
```

The saved EEG dictionaries must include the preprocessed EEG arrays and metadata
used by `load_eeg_data`, including channel names and time values when available.

## Pretrained Visual Models

Run the helper to download the common visual-model weights into
`$PRETRAIN_WEIGHTS_DIR`:

```bash
python encoding/pretrain_weights/download_pretrain_weights.py
```

DINO and DINOv2 wrappers do not vendor upstream repositories. They use
`torch.hub` by default, or a local upstream checkout when provided:

```bash
export DINO_REPO=/path/to/facebookresearch_dino
export DINOV2_REPO=/path/to/facebookresearch_dinov2
```

CORnet is also not vendored because the upstream project is GPL-licensed. Install
it separately if you need `cornet_s` experiments.

## Minimal Run

From the repository root:

```bash
python encoding/dnn_feature_maps_extraction/extract_feature_maps.py \
  --dnn alexnet \
  --pretrained true \
  --project_dir "$PROJECT_DIR" \
  --image_set_dir "$IMAGE_SET_DIR"
```

```bash
bash encoding/synthesizing_eeg_data/train.sh \
  --sub 1 \
  --dnn alexnet \
  --subjects within \
  --pretrained true \
  --layers all \
  --brain_regions occipital_parietal \
  --n_components 1000
```

To reuse existing PCA features and run evaluation only:

```bash
bash encoding/synthesizing_eeg_data/train.sh --sub 1 --dnn alexnet --skip_pca --skip_train
```

The pipeline reports the resolved data, image, PCA, and output directories before
running. Treat missing files at that stage as setup failures rather than model
failures.

## Reproducibility Notes

- Boolean CLI flags use `true/false`, `1/0`, or `yes/no`.
- Encoding outputs are written below `$PROJECT_DIR/results`.
- PCA features are written below `$PROJECT_DIR/dnn_feature_maps_*`.
- The linearizing encoding path supports configurable brain regions, subject
  mode, PCA dimensions, and evaluation time window.
- Full reproduction requires the same dataset version, subject split, visual
  weights, PCA settings, random seed, and EEG preprocessing used in the paper.

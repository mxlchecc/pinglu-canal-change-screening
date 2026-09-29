# Public Reproducibility Materials

## Purpose

This repository supports audit and reproduction of the principal experiments reported in the revised manuscript, *Transformer-Based Change Detection for Construction Change Screening in Large Linear Infrastructure Corridors*. It contains source code, reviewed final configurations, complete training histories for the formal three-seed matrix, validation threshold records, final result tables, figure files/source tables, Pinglu metadata summaries, and environment records. These records allow the reported aggregate tables to be recomputed without downloading images or model weights. The trained checkpoint files are intended for separate distribution; their public download link is not yet available.

## Package structure

- `01_Source_Code`: training, evaluation, inference, plotting, and shared model/data code.
- `02_Final_Experiment_Configs`: reviewed common settings, model definitions, and the 30-run registry.
- `03_Training_Histories`: the complete per-epoch histories and run summaries, organized by seed.
- `04_Validation_and_Thresholds`: per-run validation threshold grids, model-selection summaries, family thresholds, and the separate convergence audit.
- `05_Final_Results`: manuscript-aligned model, ablation, robustness, Pinglu screening, and engineering-overlap tables.
- `06_Figure_Source_Data`: 12 publishable figures and a 13-entry figure-to-source-data map. The archived project image is excluded with the restricted project data.
- `07_Pinglu_Metadata`: Pinglu-only scene metadata and verified processing summaries.
- `08_Environment`: software/hardware records, pretrained-snapshot metadata, checkpoint metadata, and execution coverage.
- `09_Trained_Weights`: the manifest and colocated run/calibration records for all 30 formal checkpoints. The `best.pt` weight files are not included in this Git repository; see the weight availability notes below.

## Public benchmark and subset qualification

SYSU-CD is the public labeled benchmark used for supervised training, validation, ablation, and held-out testing. The released subsets contain 12,000 training, 4,000 validation, and 4,000 test patch pairs. The original SYSU-CD publication reports that 800 source image pairs were divided at a 6:2:2 ratio before patch generation. The local mirror used here does not retain source identifiers or patch-to-source lineage, so the publication-reported source-level separation was not independently re-audited from local files.

Audited changed/unchanged pixel counts and changed-pixel proportions for the released training, validation, and test subsets are provided in `05_Final_Results/sysu_cd_subset_summary.csv`.

## Models and input

The controlled comparison includes SegFormer-B0, Swin-Tiny-UPerNet, and eight Mask2Former-Swin-Tiny factorial configurations. Every model receives a six-channel input formed by concatenating `RGB(t1) + RGB(t2)`. Validation selected Mask2Former `Difference + swap`; the resolution adapter was evaluated in the factorial matrix but was not retained in the selected configuration.

## Seeds and reporting

The formal matrix uses seeds `20260803`, `20260804`, and `20260805`. Each of the two baselines and each of the eight Mask2Former configurations has a real history, run summary, and validation-threshold record for all three seeds. Aggregate results are reported as `mean ± sample SD` across the three independent runs.

## Training protocol

- Optimizer: AdamW
- Initial learning rate: `6e-5`
- Weight decay: `0.01`
- Warm-up: `5%` of optimizer steps
- Schedule: cosine decay
- Batch size: `16`
- Maximum epochs: `15`
- Early stopping: `4` stagnant validation epochs
- Changed-class weight: `3.0`
- Loss: weighted cross-entropy + Dice
- Input size: `256 × 256`

One SegFormer-B0 run that reached its best fixed-threshold validation score at epoch 15 was repeated with a 25-epoch ceiling under the same seed and early-stopping rule. That separate convergence audit early-stopped after 13 epochs and is not substituted into the common-budget main comparison.

## Model and threshold selection

Checkpoints are selected using validation F1. The final Mask2Former configuration is selected solely by mean validation F1 across the three seeds; held-out test results do not participate in model selection. Each run's probability threshold is selected only on the validation split and is then locked for held-out testing. For Pinglu application, the three seed probability rasters are averaged first and the family threshold is the arithmetic mean of the three seed-specific validation-selected thresholds.

## Pinglu application and evidence boundary

The Pinglu application has no project-specific pixel-level reference mask. Its outputs are candidate-change and cross-model consensus screening layers, not target-domain accuracy estimates or direct construction-completion measurements. Archived engineering polygons are used only for descriptive spatial overlap and project context; they are not treated as supervised labels or ground truth.

## Restricted data

Original Gaofen imagery, processed Pinglu mosaics, and original project vector products are excluded because of project-management and third-party restrictions. The archived project image, `archived_project_sequences.jpg`, is also intentionally excluded; its entry in the 13-entry figure source map documents its provenance and does not indicate that the image is included. `06_Figure_Source_Data` therefore contains 12 publishable figures. `07_Pinglu_Metadata` contains only non-image metadata and verified processing summaries required to understand the application workflow.

## Model weights

The 30 formal checkpoints cover the two baselines and eight Mask2Former configurations for each of the three seeds. Their run records and `weight_manifest.csv` are included in `09_Trained_Weights`, but the `best.pt` files are not included. **Weight download: not yet publicly available.** A separate download link will be added when the weight archive has been deposited and verified.

The locally prepared checkpoint files contain trained tensors and documented primitive-valued configuration/selection metadata. Every state tensor is byte-identical to its original saved tensor; the manifest records the prepared file hash in `sha256` and the original saved checkpoint hash in `original_checkpoint_sha256`. Embedded architecture configurations allow offline initialization without downloading pretrained weights. Local checks of all 30 prepared files covered safe tensor loading, strict state-dictionary loading, parameter-count/epoch/seed consistency, and finite-output forward passes. These records document the local checks, not the availability of a public weight download.

Before running checkpoint inspection, supplied-weight evaluation, or Pinglu inference, obtain the weight files through the separate download once available and place each `best.pt` at the repository-relative path recorded in the manifest's `weight_file` column. Compare the downloaded file's SHA-256 digest with the `sha256` column, not `original_checkpoint_sha256`. Detailed placement and verification notes are provided in [`09_Trained_Weights/README.md`](09_Trained_Weights/README.md).

The prepared weights were selected on fixed-threshold validation F1, and each colocated threshold record contains the separately validation-selected probability threshold. Do not choose a new threshold on the test set. `mask2former_difference_swap_seed...` is the selected Mask2Former configuration.

## Reproduction entry points

Commands below are run from the repository root, whether cloned or downloaded as an archive. Python 3.12 is recommended. The recorded environment used PyTorch 2.11.0, CUDA 12.8, and an NVIDIA GeForce RTX 4060 Ti. GPU operations may differ slightly across platforms; throughput is hardware-specific.

### Install

```console
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell; `source .venv/bin/activate` in a Unix shell), then install the recorded CUDA build and other runtime dependencies:

```console
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r 01_Source_Code/requirements-runtime.txt
```

For CPU-only checkpoint inspection/evaluation, use the corresponding CPU PyTorch build. Training/calibration and Pinglu batch scripts reproduce the recorded CUDA execution path.

### Verify repository files

```console
python 01_Source_Code/scripts/verify_release.py
```

`verify_release.py` validates the files listed in this Git repository's `checksums.sha256`. That manifest covers the deposited code, documentation, and evidence files, not separately distributed checkpoint binaries or newly generated outputs.

### Inspect weights after obtaining the separate download

This step requires all 30 `best.pt` files to be present at the paths recorded in `09_Trained_Weights/weight_manifest.csv`. It cannot run from the Git repository alone while the weight download is unavailable. Verify each downloaded file against the manifest's `sha256` column before loading it, then run:

```console
python 01_Source_Code/scripts/check_weights.py --device cpu
```

`check_weights.py` safely loads the checkpoints and checks their model keys, tensor shapes, metadata, and parameter counts. Add `--forward --device cuda` for a synthetic finite-output functional check. This check is not an accuracy experiment.

### Obtain the public benchmark

```console
python 01_Source_Code/scripts/download_sysu.py data/SYSU_CD_HF
```

The pinned public mirror is `ericyu/SYSU_CD`, revision `fa1aa2f7a050b015a03417b2ac45c34506c031b8`. The data loader expects `train-*.parquet`, `val-*.parquet`, and `test-*.parquet` under `data/SYSU_CD_HF/data`. If the files are elsewhere, edit `data.sysu_root` in `02_Final_Experiment_Configs/runtime_config.yaml`; relative paths resolve from the package root. The released-subset lineage limitation stated above still applies.

### Evaluate models after obtaining the separate weights

The following example requires the benchmark download and `09_Trained_Weights/mask2former_difference_swap_seed20260803/best.pt`. It is not runnable from the Git repository alone until the separate weight file has been obtained.

```console
python 01_Source_Code/scripts/evaluate.py 09_Trained_Weights/mask2former_difference_swap_seed20260803/best.pt --config 02_Final_Experiment_Configs/runtime_config.yaml --split test --resolutions 0.5 --output outputs/mask2former_seed20260803_test.csv
```

The threshold is read automatically from the checkpoint's colocated calibration record. Repeat for the three seeds and baselines; use `--resolutions 0.5 0.8 2.0 2.3` for the resolution-degradation evaluation. `--device cpu --max-samples 2` provides a bounded functional check; omit `--max-samples` for the full experiment. Keep new outputs under `outputs/` so the archived evidence remains unchanged.

### Recompute tables from the archived run records

```console
python 01_Source_Code/scripts/aggregate_multiseed_results.py --new-root 09_Trained_Weights --output outputs/recomputed_tables
```

This recalculates mean and sample standard deviation and applies the same validation-only Mask2Former selection rule. The required histories, summaries, and threshold records are colocated in `09_Trained_Weights`; this command does not load `best.pt`. No training, dataset download, or weight download is needed for this step.

### Retrain the formal matrix

```console
python 01_Source_Code/scripts/run_multiseed_matrix.py --config 02_Final_Experiment_Configs/runtime_config.yaml --output-root outputs/retrained_runs --log-root outputs/training_logs --status outputs/matrix_status.jsonl
```

This invokes `train.py` and validation-only threshold calibration for all 30 formal runs. The runtime configuration contains executable model definitions, pinned pretrained revisions, and the original loss-key names used by the training code. It is separate from `common_training_config.yaml`, which documents the audited protocol. Retraining downloads the pinned upstream initialization checkpoints if they are not cached.

### Pinglu application with authorized project inputs

This step requires both the authorized project inputs and the nine selected-family checkpoint files from the separate weight distribution. Recomputing the tables alone does not supply those weight files.

```console
python 01_Source_Code/scripts/prepare_pinglu_tiles.py restricted_data/mosaic_a restricted_data/mosaic_b restricted_data/corridor_boundary.shp outputs/pinglu_tiles
python 01_Source_Code/scripts/run_pinglu_multiseed_application.py --results outputs/recomputed_tables --tiles outputs/pinglu_tiles --output outputs/pinglu_predictions --analysis outputs/pinglu_analysis
```

The first command creates the tile manifest/profile from the authorized mosaics and corridor. The second uses the nine selected-family checkpoints located through the recomputed table and performs input-order averaging and seed ensembling. Exact application reproduction requires the excluded project inputs; those inputs cannot be reconstructed from the public summaries. Run these commands on CUDA, with the source rasters' required sidecar files present.

## Interpretation and publication status

The three controlled model families have close benchmark scores; these materials do not establish a statistically reliable model ranking. Resolution degradation and image translation are controlled perturbation experiments, not target-domain accuracy validation. Pinglu outputs remain candidate-change screening layers. This repository contains the code and supporting result records. Public distribution of the checkpoint binaries remains pending, and no weight-download URL or archive DOI is claimed here.

Use `checksums.sha256` to verify the listed repository files and `09_Trained_Weights/weight_manifest.csv` to verify separately obtained checkpoint files. The original scene rasters, mosaics, and project vectors are excluded. The upstream model/data identifiers are documented for provenance; users must observe their respective access and reuse terms. No open-source license is assigned to the authors' code in this repository; reuse permissions must be confirmed with the rights holders.

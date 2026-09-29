# Source code

See the package-root README for installation, exact commands, and data restrictions. The executable configuration is `../02_Final_Experiment_Configs/runtime_config.yaml`. Training, calibration, and Pinglu processing follow the original CUDA execution path. Supplied-weight evaluation and checkpoint inspection additionally support CPU execution.

The 30 trained checkpoint files are available as three independent ZIP assets in [release v1.0.0](https://github.com/mxlchecc/pinglu-canal-change-screening/releases/tag/v1.0.0). Download and extract all three ZIPs into the repository root, preserving their `09_Trained_Weights/<run_name>/best.pt` paths, before inspecting or evaluating the supplied models. The archived tables can be recomputed from the run records already in Git without downloading weights or imagery.

Model definitions are in `src/models.py`; synchronized data augmentation and resolution simulation are in `src/dataset.py`; metrics and losses are in `src/metrics.py` and `src/losses.py`. Released checkpoints carry embedded architecture configuration, so inference does not require fetching upstream initialization tensors.

`scripts/aggregate_multiseed_results.py` recomputes reported summaries from the colocated histories and calibration records without loading images. Plotting and geospatial scripts accept explicit input/output arguments; inspect `--help` before running. Their default output folders are relative to the extracted package and contain only generated artifacts.

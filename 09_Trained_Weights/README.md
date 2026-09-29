# Trained weight records and availability

## Availability

This directory contains the manifest and supporting records for 30 formal trained checkpoints: SegFormer-B0, Swin-Tiny-UPerNet, and eight Mask2Former configurations, each evaluated with seeds `20260803`, `20260804`, and `20260805`.

All 30 `best.pt` checkpoint binaries are publicly available as assets of [release v1.0.0](https://github.com/mxlchecc/pinglu-canal-change-screening/releases/tag/v1.0.0):

- [Trained_Weights_seed20260803.zip](https://github.com/mxlchecc/pinglu-canal-change-screening/releases/download/v1.0.0/Trained_Weights_seed20260803.zip) — 10 checkpoints
- [Trained_Weights_seed20260804.zip](https://github.com/mxlchecc/pinglu-canal-change-screening/releases/download/v1.0.0/Trained_Weights_seed20260804.zip) — 10 checkpoints
- [Trained_Weights_seed20260805.zip](https://github.com/mxlchecc/pinglu-canal-change-screening/releases/download/v1.0.0/Trained_Weights_seed20260805.zip) — 10 checkpoints
- [SHA256SUMS.txt](https://github.com/mxlchecc/pinglu-canal-change-screening/releases/download/v1.0.0/SHA256SUMS.txt) — checksums for the three archives

These are three independent ZIP archives, not split volumes. Download all three for the complete matrix. The checkpoint binaries are stored as release assets rather than Git-tracked files, so GitHub's automatically generated **Source code (zip)** and **Source code (tar.gz)** downloads do not contain them.

Each run directory contains its training history, summary, threshold-calibration records, and weight metadata. These small files support inspection and table recomputation without loading a checkpoint. From the repository root:

```console
python 01_Source_Code/scripts/aggregate_multiseed_results.py --new-root 09_Trained_Weights --output outputs/recomputed_tables
```

## Extract the weight archives

Verify the downloaded archives against `SHA256SUMS.txt`, then extract each ZIP directly into the repository root, preserving its internal directory structure. Do not add a folder named after the ZIP. Each `best.pt` should appear at the repository-relative path in the `weight_file` column of `weight_manifest.csv`, for example:

```text
09_Trained_Weights/mask2former_difference_swap_seed20260803/best.pt
```

Keep the existing run records beside the corresponding checkpoint. Do not replace them with another seed's records or select new probability thresholds on the test set. The selected Mask2Former configuration is `difference_swap`; the family comparison uses its three seeds and the three seeds of each baseline.

## Integrity and functional checks

`SHA256SUMS.txt` provides an archive-level check before extraction. For individual checkpoints, compare the SHA-256 digest with the manifest's `sha256` value and the byte count with `weight_bytes`. The `sha256` column describes the distribution file. `original_checkpoint_sha256` identifies the earlier saved checkpoint before its metadata was made portable; it is retained for provenance and is not the expected hash of the distribution file. `state_tensor_sha256` records the trained tensor content.

For example, in PowerShell, from the repository root:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "09_Trained_Weights/mask2former_difference_swap_seed20260803/best.pt"
```

The root `checksums.sha256` verifies the code, documentation, and evidence files deposited in Git. It does not cover checkpoint binaries downloaded separately. Use `weight_manifest.csv` for those files.

After all 30 verified checkpoint binaries are present and the documented dependencies are installed, run:

```console
python 01_Source_Code/scripts/check_weights.py --device cpu
```

Optionally add `--forward --device cuda` for a synthetic finite-output forward check. Local records of safe tensor loading, strict state-dictionary loading, parameter-count/epoch/seed checks, and forward passes are provided in the manifest and per-run metadata. Functional checks are not accuracy experiments.

Benchmark evaluation requires both the relevant checkpoint and the public SYSU-CD data described in the root README. Pinglu inference additionally requires the excluded, authorized project imagery and boundary inputs. The public metadata and aggregate records do not reconstruct those restricted inputs.

Upstream model identifiers and revisions are retained in the manifest for provenance. Users must comply with the applicable upstream access and reuse terms; this directory does not grant additional third-party rights.

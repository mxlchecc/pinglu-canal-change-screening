# Trained weight records and availability

## Availability

This directory contains the manifest and supporting records for 30 formal trained checkpoints: SegFormer-B0, Swin-Tiny-UPerNet, and eight Mask2Former configurations, each evaluated with seeds `20260803`, `20260804`, and `20260805`.

The `best.pt` checkpoint binaries are **not included in this Git repository**. Their separate public download is not yet available. A download link will be added after the weight archive has been deposited and verified. The presence of a manifest or a successful local-check record does not mean the weights are already publicly downloadable.

Each run directory contains its training history, summary, threshold-calibration records, and weight metadata. These small files support inspection and table recomputation without loading a checkpoint. From the repository root:

```console
python 01_Source_Code/scripts/aggregate_multiseed_results.py --new-root 09_Trained_Weights --output outputs/recomputed_tables
```

## Where to place separately obtained weights

After the weight download becomes available, place each `best.pt` at the repository-relative path in the `weight_file` column of `weight_manifest.csv`. For example:

```text
09_Trained_Weights/mask2former_difference_swap_seed20260803/best.pt
```

Keep the existing run records beside the corresponding checkpoint. Do not replace them with another seed's records or select new probability thresholds on the test set. The selected Mask2Former configuration is `difference_swap`; the family comparison uses its three seeds and the three seeds of each baseline.

## Integrity and functional checks

For each downloaded checkpoint, compare its SHA-256 digest with the manifest's `sha256` value and its byte count with `weight_bytes`. The `sha256` column describes the locally prepared distribution file. `original_checkpoint_sha256` identifies the earlier saved checkpoint before its metadata was made portable; it is retained for provenance and is not the expected hash of the distribution file. `state_tensor_sha256` records the trained tensor content.

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

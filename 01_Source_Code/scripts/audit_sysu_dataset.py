from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    report: dict[str, object] = {"root": str(args.root.resolve()), "splits": {}}
    for split, prefix in (("train", "train"), ("validation", "val"), ("test", "test")):
        files = sorted(args.root.glob(f"{prefix}-*.parquet"))
        changed = 0
        total = 0
        samples = 0
        positive_samples = 0
        paths: list[dict[str, object]] = []
        for path in files:
            parquet = pq.ParquetFile(path)
            paths.append({"name": path.name, "rows": parquet.metadata.num_rows, "bytes": path.stat().st_size})
            for batch in tqdm(
                parquet.iter_batches(columns=["label"], batch_size=128),
                total=(parquet.metadata.num_rows + 127) // 128,
                desc=f"{split}:{path.name[:18]}",
            ):
                for value in batch.column(0).to_pylist():
                    mask = np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("L")) > 127
                    count = int(mask.sum())
                    changed += count
                    total += int(mask.size)
                    samples += 1
                    positive_samples += int(count > 0)
        report["splits"][split] = {
            "sample_pairs": samples,
            "positive_sample_pairs": positive_samples,
            "negative_sample_pairs": samples - positive_samples,
            "changed_pixels": changed,
            "unchanged_pixels": total - changed,
            "total_pixels": total,
            "changed_fraction": changed / total if total else None,
            "shards": paths,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

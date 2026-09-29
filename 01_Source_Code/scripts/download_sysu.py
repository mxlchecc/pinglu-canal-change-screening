from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY = "ericyu/SYSU_CD"
REVISION = "fa1aa2f7a050b015a03417b2ac45c34506c031b8"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the exact SYSU-CD snapshot used in the revision.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPOSITORY,
        repo_type="dataset",
        revision=REVISION,
        local_dir=args.output,
    )


if __name__ == "__main__":
    main()

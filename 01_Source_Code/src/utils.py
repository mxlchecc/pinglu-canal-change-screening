from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: Path) -> dict:
    """Resolve configured paths against the release root or environment variables."""
    import yaml
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    for section, key in (("data", "sysu_root"), ("output", "root"),
                         ("pinglu", "mosaic_a"), ("pinglu", "mosaic_b"),
                         ("pinglu", "corridor_shapefile")):
        if key not in config.get(section, {}):
            continue
        raw = os.path.expandvars(str(config[section][key]))
        if "$" in raw:
            raise ValueError(f"Unresolved environment variable in {section}.{key}: {raw}")
        value = Path(raw).expanduser()
        config[section][key] = str(value if value.is_absolute() else root / value)
    return config

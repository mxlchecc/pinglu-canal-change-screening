from __future__ import annotations
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[2]
count = 0
for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
    expected, name = line.split("  ", 1)
    path = root / name
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    if h.hexdigest() != expected:
        raise SystemExit(f"Checksum mismatch: {name}")
    count += 1
print(f"Verified {count} files.")

"""Verify the published source tree against SHA256SUMS."""
from pathlib import Path
import hashlib


def main():
    root = Path(__file__).resolve().parents[1]
    count = 0
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes source tree")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source checksum mismatch: {name}")
        count += 1
    print(f"Verified {count} source files.")


if __name__ == "__main__":
    main()

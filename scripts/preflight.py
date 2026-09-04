#!/usr/bin/env python3
"""Validate runtime versions and required local training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    "walk policy": ROOT / "models/model_walk.pt",
    "stand policy": ROOT / "models/model_stand.pt",
    "teacher checkpoint": ROOT / "move_garbage/checkpoints/model_step_012900.pt",
    "checkpoint config": ROOT / "move_garbage/checkpoints/config.yaml",
    "G1 asset": ROOT / "gr00t/rl/data/robots/g1/g1_43dof_s3.usda",
    "dynamic bin asset": (
        ROOT
        / "move_garbage/data/objects/bin/RecyclingContainerWithWheels_A01_Blue_01_dynamic.usd"
    ),
    "static bin asset": (
        ROOT
        / "move_garbage/data/objects/bin/RecyclingContainerWithWheels_A01_Blue_01_static.usd"
    ),
}

EXPECTED_HASHES = {
    "models/model_walk.pt": "2efa9d39c6ace247d6d68b43452f591cd5b22fb2f5e58dcfd3b6c5d1b30df77e",
    "models/model_stand.pt": "ceb976d2745bdaa99e51ba7141e441dc7a140638ff8e2b0efaf50efd20290054",
    "move_garbage/checkpoints/model_step_012900.pt": (
        "f148b48925d57dea4a1ff277e5a272ce7d98b21caac24613f171f8f46666e1ed"
    ),
    "move_garbage/data/objects/bin/RecyclingContainerWithWheels_A01_Blue_01_dynamic.usd": (
        "b7bbad4bd66707200e94e64afddb15a86c26617d6f235968c9a4c32dcdcb2c44"
    ),
    "move_garbage/data/objects/bin/RecyclingContainerWithWheels_A01_Blue_01_static.usd": (
        "181dbab644bfd1ed720abfdada15c287079ceb1dd84288ff9834648bfa67c32b"
    ),
}

EXPECTED_VERSIONS = {
    "accelerate": "1.13.0",
    "datasets": "5.0.1",
    "gymnasium": "1.2.0",
    "hydra-core": "1.3.2",
    "isaaclab": "2.3.2.post1",
    "numpy": "1.26.4",
    "omegaconf": "2.3.0",
    "pandas": "3.0.5",
    "torch": "2.7.0",
    "transformers": "5.12.1",
    "trl": "0.17.0",
    "wandb": "0.26.1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-packages",
        action="store_true",
        help="Validate files without checking installed package versions.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    for label, path in REQUIRED_FILES.items():
        if not path.is_file():
            errors.append(f"missing {label}: {path.relative_to(ROOT)}")

    for relative_path, expected in EXPECTED_HASHES.items():
        path = ROOT / relative_path
        if path.is_file() and sha256(path) != expected:
            errors.append(f"hash mismatch: {relative_path}")

    robot_asset = REQUIRED_FILES["G1 asset"]
    if robot_asset.is_file() and "://" in robot_asset.read_text(errors="replace"):
        errors.append("G1 asset still contains a remote reference")

    simulator_api = ROOT / "gr00t/rl/simulator/isaacsim/.isaacsim_version"
    if not simulator_api.is_file() or simulator_api.read_text().strip() != "4.5":
        errors.append("unexpected Isaac Sim compatibility marker (expected 4.5)")

    if not args.skip_packages:
        for package, expected in EXPECTED_VERSIONS.items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"missing package: {package}=={expected}")
                continue
            if not actual.startswith(expected):
                errors.append(f"version mismatch: {package} expected {expected}, found {actual}")

    if errors:
        print("Preflight failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

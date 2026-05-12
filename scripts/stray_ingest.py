#!/usr/bin/env python3
"""Preprocess a Stray Scanner recording and emit configs/stray/<scene>.yaml.

Examples:
  python scripts/stray_ingest.py dde4a31c36
  python scripts/stray_ingest.py --all
  python scripts/stray_ingest.py my_scene --resolution 640 480 --confidence-threshold 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_intrinsic(path: Path) -> tuple[float, float, float, float]:
    with path.open() as f:
        row0 = f.readline().split()
        row1 = f.readline().split()
    fx = float(row0[0])
    cx = float(row0[2])
    fy = float(row1[1])
    cy = float(row1[2])
    return fx, fy, cx, cy


def _write_scene_config(
    scene: str,
    processed_dir: Path,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    config_path: Path,
) -> None:
    root = _repo_root()
    rel = os.path.relpath(processed_dir, root).replace(os.sep, "/")
    text = f"""inherit_from: "configs/stray/base_config.yaml"

Dataset:
  sequence_name: "{scene}"
  dataset_path: "{rel}"
  Calibration:
    fx: {fx:.2f}
    fy: {fy:.2f}
    cx: {cx:.2f}
    cy: {cy:.2f}
    k1: 0.0
    k2: 0.0
    p1: 0.0
    p2: 0.0
    k3: 0.0
    distorted: False
    width: {width}
    height: {height}
    depth_scale: 1.0
    depth_noise_scale: 0.02
"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text)


def _is_stray_recording(raw_dir: Path) -> bool:
    return (
        (raw_dir / "rgb.mp4").is_file()
        and (raw_dir / "odometry.csv").is_file()
        and (raw_dir / "depth").is_dir()
    )


def ingest_scene(
    scene: str,
    *,
    repo: Path,
    width: int,
    height: int,
    confidence: int,
    skip_preprocess: bool,
) -> int:
    raw = repo / "datasets" / "stray_raw" / scene
    out = repo / "datasets" / "stray" / scene
    if not raw.is_dir():
        print(f"Error: raw directory not found: {raw}", file=sys.stderr)
        return 1
    if not skip_preprocess and not _is_stray_recording(raw):
        print(f"Error: incomplete Stray recording (need rgb.mp4, odometry.csv, depth/): {raw}", file=sys.stderr)
        return 1

    prep = repo / "scripts" / "stray_preprocess.py"
    if not skip_preprocess:
        cmd = [
            sys.executable,
            str(prep),
            "--input",
            str(raw),
            "--output",
            str(out),
            "--resolution",
            str(width),
            str(height),
            "--confidence-threshold",
            str(confidence),
        ]
        r = subprocess.run(cmd, cwd=repo)
        if r.returncode != 0:
            return r.returncode

    intrinsic = out / "intrinsic.txt"
    if not intrinsic.is_file():
        print(f"Error: missing {intrinsic} (run without --config-only first)", file=sys.stderr)
        return 1

    fx, fy, cx, cy = _parse_intrinsic(intrinsic)
    cfg = repo / "configs" / "stray" / f"{scene}.yaml"
    _write_scene_config(scene, out, fx, fy, cx, cy, width, height, cfg)
    print(f"Wrote {cfg}")
    print(f"Run: python slam.py --config {cfg.relative_to(repo)} -v")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Preprocess Stray Scanner data and write a scene YAML.")
    p.add_argument(
        "scene",
        nargs="?",
        help="Folder name under datasets/stray_raw/ (ignored if --all)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Process every subdirectory of datasets/stray_raw/ that looks like a recording",
    )
    p.add_argument("--resolution", nargs=2, type=int, default=[640, 480], metavar=("W", "H"))
    p.add_argument("--confidence-threshold", type=int, default=0, choices=[0, 1, 2])
    p.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Only (re)write the YAML from existing datasets/stray/<scene>/intrinsic.txt",
    )
    p.add_argument(
        "--config-only",
        action="store_true",
        help="Same as --skip-preprocess",
    )
    args = p.parse_args()
    repo = _repo_root()
    w, h = args.resolution
    skip = args.skip_preprocess or args.config_only

    if args.all:
        raw_root = repo / "datasets" / "stray_raw"
        if not raw_root.is_dir():
            print(f"Error: {raw_root} does not exist", file=sys.stderr)
            return 1
        codes = []
        for d in sorted(raw_root.iterdir(), key=lambda x: x.name):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if not _is_stray_recording(d):
                continue
            rc = ingest_scene(
                d.name,
                repo=repo,
                width=w,
                height=h,
                confidence=args.confidence_threshold,
                skip_preprocess=skip,
            )
            codes.append(rc)
        if not codes:
            print("No complete Stray recordings found under datasets/stray_raw/", file=sys.stderr)
            return 1
        return 0 if all(c == 0 for c in codes) else 1

    if not args.scene:
        p.error("provide SCENE name or use --all")
    return ingest_scene(
        args.scene,
        repo=repo,
        width=w,
        height=h,
        confidence=args.confidence_threshold,
        skip_preprocess=skip,
    )


if __name__ == "__main__":
    sys.exit(main())

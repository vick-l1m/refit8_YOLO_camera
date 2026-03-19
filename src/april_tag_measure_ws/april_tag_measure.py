#!/usr/bin/env python3
"""
April_tag_measure.py

Interactive AprilTag ruler.

- Purpose: Detect an AprilTag in an image and let the user measure distances by drawing lines.
- Inputs: image path, tag family, tag physical size, optional camera intrinsics
- Outputs: annotated image + JSON results
- Assumptions:
    * Measurements are on the same plane as the detected tag.
    * With intrinsics: uses pose + ray/plane intersection (more accurate under perspective).
    * Without intrinsics: uses pixel-to-meter scale from the tag edge length (approx).
- Controls:
    * Mouse: Left-click drag to draw a measurement line
    * Keys: r=reset lines, s=save + exit, q/ESC=quit

Date: 5/03/2026
Version: 1.0
Maintainer: Victor Lim - victor@polymaya.tech
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

import cv2


from src.april_tag_measure_ws.april_tags import detect_apriltags, avg_tag_edge_px
from src.april_tag_measure_ws.measurements import build_camera_matrix, load_intrinsics_json
from src.april_tag_measure_ws.interactive_ruler import InteractiveRuler


def print_header(args: argparse.Namespace) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n=== AprilTag Ruler ===")
    print(f"Time:            {now}")
    print(f"Python:          {sys.version.split()[0]}")
    print(f"Platform:        {platform.platform()}")
    print(f"OpenCV:          {cv2.__version__}")
    print(f"Image:           {args.image}")
    print(f"Tag family:      {args.family}")
    print(f"Tag size (m):    {args.tag_size_m}")
    if args.intrinsics:
        print(f"Intrinsics file: {args.intrinsics}")
    elif args.fx > 0 and args.fy > 0:
        print(f"Intrinsics:      fx={args.fx} fy={args.fy} cx={args.cx} cy={args.cy}")
    else:
        print("Intrinsics:      (none) -> quick scale mode")
    print(f"Output dir:      {args.outdir}")
    print(f"Output name:     {args.name}")
    print("======================\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="AprilTag-based interactive ruler.")
    ap.add_argument("image", type=str, help="Path to image.jpg")
    ap.add_argument("--tag-size-m", type=float, required=True,
                    help="Physical AprilTag size (outer black square) in meters")
    ap.add_argument("--family", type=str, default="tag25h9",
                    help="AprilTag family (e.g. tag25h9, tag36h11)")

    ap.add_argument("--intrinsics", type=str, default="",
                    help="Path to intrinsics JSON (fx,fy,cx,cy, optional dist)")
    ap.add_argument("--fx", type=float, default=0.0)
    ap.add_argument("--fy", type=float, default=0.0)
    ap.add_argument("--cx", type=float, default=0.0)
    ap.add_argument("--cy", type=float, default=0.0)

    # Default outputs to <repo_root>/captures/april_tag_measure
    default_outdir = (Path(__file__).resolve().parents[2] / "captures" / "april_tag_measure")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output directory")
    
    ap.add_argument("--name", type=str, default="apriltag_ruler", help="Basename for outputs")

    args = ap.parse_args()
    print_header(args)

    repo_root = Path(__file__).resolve().parents[2]
    
    # Ensure intrinsics path is absolute (if provided)
    if args.intrinsics:
        intr_path = Path(args.intrinsics)
        if not intr_path.is_absolute():
            # Try relative to repo root first
            candidate = repo_root / intr_path
            if candidate.exists():
                args.intrinsics = str(candidate)
                print(f"Resolved intrinsics path to: {args.intrinsics}")
                
    img_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dets = detect_apriltags(gray, args.family)
    if not dets:
        raise SystemExit("No AprilTags detected. Check family, lighting, tag size in pixels.")

    # Choose largest tag (most reliable)
    dets_sorted = sorted(dets, key=lambda d: avg_tag_edge_px(d["corners"]), reverse=True)
    tag = dets_sorted[0]
    print(f"Detected {len(dets_sorted)} tag(s). Using ID={tag['id']} backend={tag.get('backend')}")

    # Intrinsics (optional)
    K = None
    dist = None
    if args.intrinsics:
        K, dist = load_intrinsics_json(Path(args.intrinsics))
    elif args.fx > 0 and args.fy > 0:
        K = build_camera_matrix(args.fx, args.fy, args.cx, args.cy)
        dist = None

    ruler = InteractiveRuler(img, tag, args.family, args.tag_size_m, K, dist)
    res = ruler.run()

    # Save outputs (always at end)
    res.image = str(img_path)

    annotated_path = outdir / f"{args.name}_annotated.jpg"
    json_path = outdir / f"{args.name}_results.json"

    cv2.imwrite(str(annotated_path), ruler.img)
    json_path.write_text(json.dumps(asdict(res), indent=2))

    print("Saved:", annotated_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
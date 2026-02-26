#!/usr/bin/env python3
"""
calibrate_camera.py

OpenCV checkerboard calibration script (reusable across cameras).

Outputs a JSON file with:
- camera_matrix (fx, fy, cx, cy)
- dist_coeffs
- reprojection RMSE
"""

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="OpenCV checkerboard camera calibration.")
    p.add_argument("--images", required=True, help="Glob, e.g. 'calib_imgs/*.jpg'")
    p.add_argument("--rows", type=int, required=True, help="Inner corners rows (e.g., 6)")
    p.add_argument("--cols", type=int, required=True, help="Inner corners cols (e.g., 9)")
    p.add_argument("--square", type=float, required=True, help="Square size in meters (e.g., 0.025)")
    p.add_argument("--output", default="intrinsics.json", help="Output JSON path")
    p.add_argument("--visualize", action="store_true", help="Show detected corners")
    return p.parse_args()


def main():
    args = parse_args()
    img_paths = sorted(glob.glob(args.images))
    if not img_paths:
        raise SystemExit(f"No images matched: {args.images}")

    pattern_size = (args.cols, args.rows)  # (cols, rows)
    square = float(args.square)

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= square

    objpoints = []
    imgpoints = []
    img_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    used = 0
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping unreadable image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w,h)

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            print(f"Not found: {path}")
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        used += 1

        if args.visualize:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, True)
            cv2.imshow("Corners", vis)
            cv2.waitKey(50)

    if args.visualize:
        cv2.destroyAllWindows()

    if used < 10:
        raise SystemExit(f"Only {used} usable images. Take more (aim 20–40).")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # Reprojection error (RMSE in pixels)
    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        total_error += err ** 2
        total_points += len(objpoints[i])
    rmse = (total_error / total_points) ** 0.5

    out = {
        "image_width": img_size[0],
        "image_height": img_size[1],
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "reprojection_rmse_px": float(rmse),
        "checkerboard": {
            "inner_corners_rows": args.rows,
            "inner_corners_cols": args.cols,
            "square_size_m": square
        }
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")
    print(f"Reprojection RMSE (px): {rmse:.4f}")
    print("camera_matrix:\n", camera_matrix)
    print("dist_coeffs:\n", dist_coeffs)


if __name__ == "__main__":
    main()
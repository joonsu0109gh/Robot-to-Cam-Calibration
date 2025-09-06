#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate T_cam2base by projecting EE axes onto images.

"""

import os
import json
import glob
import argparse
import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R


# -------------------------------
# Config loader
# -------------------------------
def load_config(config_path: str) -> dict:
    """Load config.yaml and return output_root_dir."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    out_root = cfg.get("output_root_dir", "./output")
    return {"output_root_dir": out_root}


# -------------------------------
# Original helper functions 
# -------------------------------
def load_intrinsics(json_path):
    """Load intrinsics (K, dist) from RealSense SDK JSON saved earlier."""
    with open(json_path, "r") as f:
        j = json.load(f)
    K = np.array([
        [j["fx"], 0,        j["ppx"]],
        [0,        j["fy"], j["ppy"]],
        [0,        0,       1]
    ], dtype=np.float64)
    # Distortion coeffs can be (N,), reshape to (N,1)
    D = np.array(j.get("coeffs", []), dtype=np.float64).reshape(-1, 1)
    return K, D


def project_points(points_3d_in_base, K, D, T_cam2base):
    """
    Project 3D points (in BASE frame) to image using K, D, and cam2base.
    """
    # homogeneous points
    num_pts = points_3d_in_base.shape[0]
    points_3d_hom = np.hstack([points_3d_in_base, np.ones((num_pts, 1), dtype=np.float64)])
    T_base2cam = np.linalg.inv(T_cam2base)
    points_3d_in_cam = (T_base2cam @ points_3d_hom.T).T[:, :3]  # (N,3)

    # project points without cv2
    img_pts, _ = cv2.projectPoints(
        points_3d_in_cam,
        rvec=np.zeros((3, 1), dtype=np.float64),  # no rotation needed
        tvec=np.zeros((3, 1), dtype=np.float64),  # no translation needed
        cameraMatrix=K,
        distCoeffs=D
    )
    return img_pts.reshape(-1, 2)  # (N,2)


def make_homogeneous_coord(point_3d):
    """Convert 3 to 4x1 homogeneous coordinates."""
    return np.hstack([point_3d, 1])  # (4,1)


def make_ee_axes_points_in_base(T_ee2base, axis_len=0.05):
    """
    Create 4 points in BASE frame: EE origin and endpoints of X/Y/Z axes.
    """
    p0 = T_ee2base[:3, 3]
    px = (T_ee2base @ make_homogeneous_coord(np.array([1, 0, 0], dtype=np.float64) * axis_len))
    py = (T_ee2base @ make_homogeneous_coord(np.array([0, 1, 0], dtype=np.float64) * axis_len))
    pz = (T_ee2base @ make_homogeneous_coord(np.array([0, 0, 1], dtype=np.float64) * axis_len))
    return np.stack([p0, px[:3], py[:3], pz[:3]], axis=0)  # (4,3)


def draw_axes_on_image(img_bgr, pts2d, thickness=2):
    """
    Draw EE axes on image. pts2d: 4x2 [origin, X, Y, Z].
    OpenCV convention: We'll draw X(red), Y(green), Z(blue).
    """
    vis = img_bgr.copy()
    o = tuple(np.round(pts2d[0]).astype(int))
    x = tuple(np.round(pts2d[1]).astype(int))
    y = tuple(np.round(pts2d[2]).astype(int))
    z = tuple(np.round(pts2d[3]).astype(int))

    # origin
    cv2.circle(vis, o, 5, (0, 0, 0), -1)

    # axes
    cv2.line(vis, o, x, (0, 0, 255), thickness)  # X - red (BGR)
    cv2.line(vis, o, y, (0, 255, 0), thickness)  # Y - green
    cv2.line(vis, o, z, (255, 0, 0), thickness)  # Z - blue

    # labels
    cv2.putText(vis, "X", x, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis, "Y", y, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis, "Z", z, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return vis


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Project EE axes onto images.")
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    ap.add_argument(
        "-i", "--intrinsics_filename",
        type=str,
        default=None,
        help="Filename inside <output_root_dir>/camera used to use estimated intrinsics (default: estimated_rgb_intrinsics.json)."
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ----------------- Paths from config -----------------
    out_root = cfg["output_root_dir"]

    # Data roots 
    robot_dir      = os.path.join(out_root, "robot")
    detections_dir = os.path.join(robot_dir, "detections")
    images_dir     = os.path.join(robot_dir, "images")
    vis_dir        = os.path.join(robot_dir, "proj_vis")
    os.makedirs(vis_dir, exist_ok=True)

    # Intrinsics sources
    sdk_intr_json  = os.path.join(robot_dir, "realsense_rgb_sdk_intrinsics.json")  # SDK dump path
    calib_dir      = os.path.join(out_root, "camera")
    if args.intrinsics_filename is not None:
        intrinsic_path = os.path.join(calib_dir, args.intrinsics_filename)            # estimated K

    # cam2base result (from eye-to-hand solve)
    cam2base_path  = os.path.join(robot_dir, "cam2base_calibration.json")
    # -----------------------------------------------------

    # Load camera intrinsics
    K, D = load_intrinsics(sdk_intr_json)

    if args.intrinsics_filename is not None:
        with open(intrinsic_path, "r") as f:
            j = json.load(f)
        K = np.array(j["K"], dtype=np.float64).reshape(3, 3)
        D = np.array(j.get("D"), dtype=np.float64).reshape(-1, 1)  # can be empty
        print(f"[Info] Loaded intrinsics K:\n{K}\nD:\n{D.T}")

    # Load T_cam2base
    with open(cam2base_path, "r") as f:
        data = json.load(f)

    T_cam2base = np.array(data["T_cam2base"], dtype=np.float64).reshape(4, 4)
    print(f"[Info] Camera2Base:\n{T_cam2base}")

    # Enumerate detection jsons
    json_files = sorted(glob.glob(os.path.join(detections_dir, "*.json")))
    if not json_files:
        print("No detection JSONs found.")
        return

    AXIS_LEN_M = 0.06  # for visualization

    for jf in json_files:
        with open(jf, "r") as f:
            j = json.load(f)

        img_name = j.get("image_filename")
        if not img_name:
            idx = j.get("image_index", None)
            if idx is None:
                print(f"[Skip] {os.path.basename(jf)} has no image reference.")
                continue
            img_name = f"realsense_rgb_{int(idx):04d}.png"

        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[Skip] image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[Skip] failed to read image: {img_path}")
            continue

        # Build base_T_ee from JSON (expects ee_position + ee_orientation (xyzw) )
        ee_pos = j.get("ee_position", None)
        ee_quat = j.get("ee_orientation", None)
        if (ee_pos is None) or (ee_quat is None):
            print(f"[Skip] EE pose missing in {os.path.basename(jf)}")
            continue

        T_ee2base = np.eye(4, dtype=np.float64)
        T_ee2base[:3, :3] = R.from_quat(ee_quat).as_matrix()
        T_ee2base[:3, 3] = np.asarray(ee_pos, dtype=np.float64).reshape(3)

        # Build EE axes points in BASE frame
        pts3d_base = make_ee_axes_points_in_base(T_ee2base, axis_len=AXIS_LEN_M)

        # Project to image
        pts2d = project_points(pts3d_base, K, D, T_cam2base)

        # Draw and save
        vis = draw_axes_on_image(img, pts2d, thickness=2)

        # (Optional) draw board axes if rvec/tvec exist in the JSON
        ch_pose = j.get("charuco_pose", {})
        rvec = np.array(ch_pose.get("rvec", []), dtype=np.float64).reshape(-1)
        tvec = np.array(ch_pose.get("tvec", []), dtype=np.float64).reshape(-1)
        if rvec.size == 3 and tvec.size == 3:
            try:
                cv2.drawFrameAxes(
                    vis, K, D,
                    rvec.reshape(3, 1), tvec.reshape(3, 1),
                    AXIS_LEN_M
                )
            except Exception as e:
                cv2.putText(vis, f"[board draw err] {str(e)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out_name = os.path.splitext(os.path.basename(jf))[0] + "_proj.png"
        out_path = os.path.join(vis_dir, out_name)
        cv2.imwrite(out_path, vis)
        print(f"[Saved] {out_path}")

    print("[Done] Projection validation finished.")


if __name__ == "__main__":
    main()

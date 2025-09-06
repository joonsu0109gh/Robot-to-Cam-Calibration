#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Capture RGB frames from an RealSense camera for ChArUco calibration with robot poses.

Keys:
- SPACE: save one sample (only if corners >= threshold)
- 'a'  : toggle auto-capture-on-detection
- 'q'/ESC: quit

Outputs:
  <out_root>/robot/images/<series>_####.png
  <out_root>/robot/detections/<series>_####_detections.json
  <out_root>/realsense_rgb_sdk_intrinsics.json
"""

import os
import time
import json
import argparse
import numpy as np
import pyrealsense2 as rs
import cv2
import yaml

from franky import Robot
from scipy.spatial.transform import Rotation as R


# =========================
# Config loader
# =========================
def get_aruco_dict_by_name(name: str):
    """Resolve cv2.aruco predefined dictionary from its name string."""
    if not hasattr(cv2.aruco, name):
        raise ValueError(
            f"Unknown ArUco dictionary name '{name}'. "
            f"Check cv2.aruco.* constants (e.g., DICT_5X5_100)."
        )
    dict_id = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(dict_id), dict_id


def load_config(path: str) -> dict:
    """
    Load YAML config and apply defaults for optional sections.

    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    out_root = cfg.get("output_root_dir", "./output")

    # board
    board_cfg = cfg.get("board", {}) or {}
    dict_name    = board_cfg.get("dict_name", "DICT_5X5_100")
    squares_x    = int(board_cfg.get("squares_x", 5))
    squares_y    = int(board_cfg.get("squares_y", 5))
    square_len_m = float(board_cfg.get("square_len_mm", 35.0)) / 1000.0
    marker_len_m = float(board_cfg.get("marker_len_mm", 28.0)) / 1000.0
    _, dict_id   = get_aruco_dict_by_name(dict_name)

    # camera
    cam = cfg.get("camera", {}) or {}
    fps               = int(cam.get("fps", 30))
    width             = int(cam.get("width", 1280))
    height            = int(cam.get("height", 720))
    exposure_settle_s = float(cam.get("exposure_settle_sec", 1.0))
    min_corners       = int(cam.get("min_corners_to_save", 12))
    use_refinement    = bool(cam.get("use_refinement", True))
    series_name       = str(cam.get("series_name", "realsense_rgb"))

    # robot
    r = cfg.get("robot", {}) or {}
    hostname = str(r.get("hostname", "172.16.0.2"))

    return {
        "output_root_dir": out_root,
        "board": {
            "dict_id": dict_id,
            "dict_name": dict_name,
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_len_m": square_len_m,
            "marker_len_m": marker_len_m,
        },
        "camera": {
            "fps": fps,
            "width": width,
            "height": height,
            "exposure_settle_sec": exposure_settle_s,
            "min_corners_to_save": min_corners,
            "use_refinement": use_refinement,
            "series_name": series_name,
        },
        "robot": {"hostname": hostname},
    }


# =========================
# ChArUco utilities
# =========================
def build_charuco_board(dict_id, sx, sy, s_len_m, m_len_m):
    """Build a ChArUco board matching your print."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    try:
        board = cv2.aruco.CharucoBoard((sx, sy), s_len_m, m_len_m, aruco_dict)
    except Exception:
        board = cv2.aruco.CharucoBoard_create(sx, sy, s_len_m, m_len_m, aruco_dict)
    return aruco_dict, board


def detect_charuco(img_bgr, aruco_dict, board, use_refine=True):
    """Detect ArUco -> refine -> interpolate ChArUco on a BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detector (new API if available)
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    except Exception:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0:
        return 0, None, None, gray

    # Optional refinement
    if use_refine:
        try:
            corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(
                image=gray, board=board, detectedCorners=corners, detectedIds=ids,
                rejectedCorners=rejected, parameters=params
            )
        except Exception:
            pass

    # Interpolate ChArUco
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners, markerIds=ids, image=gray, board=board
    )
    n = int(retval) if retval is not None else 0
    return n, charuco_corners, charuco_ids, gray


def estimate_charuco_pose(ch_corners, ch_ids, board, camera_matrix, dist_coeffs=None):
    """
    Estimate ChArUco board pose using detected corners (camera frame).
    Returns (rvec, tvec) in Rodrigues (3x1) and meters.
    """
    if ch_corners is None or ch_ids is None or len(ch_corners) < 4:
        return None, None

    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charucoCorners=ch_corners,
        charucoIds=ch_ids,
        board=board,
        cameraMatrix=K,
        distCoeffs=D,
        rvec=np.zeros((3, 1), dtype=np.float32),
        tvec=np.zeros((3, 1), dtype=np.float32),
    )
    if retval:
        return rvec, tvec
    return None, None


def draw_overlays(
    img_bgr,
    corners,
    ids,
    n_charuco,
    rvec=None,
    tvec=None,
    camera_matrix=None,
    dist_coeffs=None,
    axis_len_m: float = None,
    idx: int = None,
    min_corners: int = 12,
):
    """Draw visual overlays for the ChArUco capture UI."""
    vis = img_bgr.copy()

    # Draw detected ChArUco corners
    try:
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
    except Exception:
        pass

    # HUD
    cv2.putText(
        vis,
        f"ChArUco corners: {n_charuco}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if n_charuco >= min_corners else (0, 0, 255),
        2,
    )
    cv2.putText(vis, "SPACE: save  |  A: auto-capture toggle", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "Q/ESC: quit", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if idx is not None:
        cv2.putText(vis, f"Capture Index: {idx}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Optional axes
    if rvec is not None and tvec is not None and camera_matrix is not None and dist_coeffs is not None:
        try:
            rvec_cv = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec_cv = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
            D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
            axis_len = float(axis_len_m if axis_len_m is not None else 0.05)
            cv2.drawFrameAxes(vis, K, D, rvec_cv, tvec_cv, axis_len)
        except Exception as e:
            cv2.putText(vis, f"[pose draw err] {str(e)}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


# =========================
# Eye-to-hand: detections record
# =========================
def make_detection_record(robot, rvec, tvec, ch_corners, ch_ids, idx):
    """Bundle robot EE pose, ChArUco pose & corners into a JSON-able dict."""
    # Franka: state.O_T_EE is 16 values row-major → 4x4 SE(3)
    ee_pose = robot.state.O_T_EE
    ee_pos = ee_pose.translation
    ee_quat_xyzw = ee_pose.quaternion
    T_ee = np.eye(4, dtype=np.float64)
    T_ee[:3, :3] = R.from_quat([ee_quat_xyzw[0], ee_quat_xyzw[1], ee_quat_xyzw[2], ee_quat_xyzw[3]]).as_matrix()
    T_ee[:3, 3] = ee_pos

    charuco_pose_block = {"rvec": [], "tvec": [], "R": [], "quat_xyzw": [], "T_se3": []}
    if rvec is not None and tvec is not None:
        R_mat, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        quat_xyzw = R.from_matrix(R_mat).as_quat()
        T_se3 = np.eye(4, dtype=np.float64)
        T_se3[:3, :3] = R_mat
        T_se3[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)

        charuco_pose_block = {
            "rvec": np.asarray(rvec, dtype=np.float64).reshape(-1).tolist(),
            "tvec": np.asarray(tvec, dtype=np.float64).reshape(-1).tolist(),
            "R": R_mat.tolist(),
            "quat_xyzw": quat_xyzw.tolist(),
            "T_se3": T_se3.tolist(),  # T_board2cam
        }

    ch_corners_list = ch_corners.reshape(-1, 2).tolist() if ch_corners is not None else []
    ch_ids_list = ch_ids.reshape(-1).astype(int).tolist() if ch_ids is not None else []

    return {
        "image_index": int(idx),
        "T_ee2base": T_ee.tolist(),
        "ee_position": ee_pos.tolist(),
        "ee_orientation": ee_quat_xyzw.tolist(),
        "charuco_corners_px": ch_corners_list,
        "charuco_ids": ch_ids_list,
        "charuco_pose": charuco_pose_block,
    }


# =========================
# CLI / Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="Eye-to-hand capture with ChArUco.")
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_root = os.path.join(cfg["output_root_dir"], "robot")
    b = cfg["board"]
    c = cfg["camera"]
    r = cfg["robot"]

    # Paths (no timestamped tree)
    IMG_DIR = os.path.join(out_root, "images")
    DET_DIR = os.path.join(out_root, "detections")
    SDK_JSON_PATH = os.path.join(out_root, "realsense_rgb_sdk_intrinsics.json")
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(DET_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SDK_JSON_PATH), exist_ok=True)

    print("[Info] Waiting for robot connection...")
    # Robot controller
    robot = Robot(r["hostname"])

    # Build board
    aruco_dict, board = build_charuco_board(
        b["dict_id"], b["squares_x"], b["squares_y"], b["square_len_m"], b["marker_len_m"]
    )
    
    print("[Info] Waiting for camera to start...")

    # Configure RealSense
    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, c["width"], c["height"], rs.format.bgr8, c["fps"])
    profile = pipeline.start(rs_cfg)

    # Dump SDK intrinsics once (overwrite each run to keep in sync)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    sdk_intr_json = {
        "width": intr.width, "height": intr.height,
        "ppx": intr.ppx, "ppy": intr.ppy, "fx": intr.fx, "fy": intr.fy,
        "model": str(intr.model), "coeffs": list(intr.coeffs)
    }
    with open(SDK_JSON_PATH, "w") as f:
        json.dump(sdk_intr_json, f, indent=2)
    print(f"[Info] Saved SDK intrinsics -> {SDK_JSON_PATH}")

    print("[Info] Pipeline started. Waiting to stabilize exposure...")
    time.sleep(c["exposure_settle_sec"])

    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array(intr.coeffs, dtype=np.float64).reshape(-1, 1)

    idx = 0
    series = c["series_name"]
    auto_capture = False  # toggled by 'a'

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())

            # Detect ChArUco and estimate pose (camera_T_board)
            n, ch_corners, ch_ids, gray = detect_charuco(
                frame_bgr, aruco_dict, board, use_refine=c["use_refinement"]
            )
            rvec, tvec = estimate_charuco_pose(ch_corners, ch_ids, board, K, D)

            vis = draw_overlays(
                frame_bgr, ch_corners, ch_ids, n,
                rvec, tvec, K, D,
                axis_len_m=b["square_len_m"], idx=idx, min_corners=c["min_corners_to_save"]
            )

            # Display
            cv2.imshow("RealSense RGB (ChArUco capture)", vis)
            key = cv2.waitKey(1) & 0xFF

            # Auto-capture when enough corners are seen
            if auto_capture and n >= c["min_corners_to_save"]:
                img_name = f"{series}_{idx:04d}.png"
                det_name = f"{series}_{idx:04d}_detections.json"

                cv2.imwrite(os.path.join(IMG_DIR, img_name), frame_bgr)
                det = make_detection_record(robot, rvec, tvec, ch_corners, ch_ids, idx)
                with open(os.path.join(DET_DIR, det_name), "w") as f:
                    json.dump(det, f, indent=2)

                print(f"[Saved] {img_name} (corners={n})")
                idx += 1
                time.sleep(0.2)  # debounce

            # Keyboard controls
            if key == ord('a'):
                auto_capture = not auto_capture
                print(f"[Toggle] auto-capture = {auto_capture}")

            elif key == ord(' ') and n >= c["min_corners_to_save"]:
                img_name = f"{series}_{idx:04d}.png"
                det_name = f"{series}_{idx:04d}_detections.json"

                cv2.imwrite(os.path.join(IMG_DIR, img_name), frame_bgr)
                det = make_detection_record(robot, rvec, tvec, ch_corners, ch_ids, idx)
                with open(os.path.join(DET_DIR, det_name), "w") as f:
                    json.dump(det, f, indent=2)

                print(f"[Saved] {img_name} (corners={n})")
                idx += 1

            elif key in (ord('q'), 27):  # 'q' or ESC
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[Done] Capture finished.")


if __name__ == "__main__":
    main()

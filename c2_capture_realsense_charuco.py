#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Capture RGB frames from an Intel RealSense camera for ChArUco calibration.
"""

import os
import time
import json
import argparse
import numpy as np
import pyrealsense2 as rs
import cv2
import yaml


# -------------------------------
# Config loading
# -------------------------------
def get_aruco_dict_by_name(name: str):
    """Resolve cv2.aruco predefined dictionary from its name string."""
    if not hasattr(cv2.aruco, name):
        raise ValueError(
            f"Unknown ArUco dictionary name '{name}'. "
            f"Check cv2.aruco.* constants (e.g., DICT_5X5_100)."
        )
    dict_id = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(dict_id), dict_id

def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    out_root = cfg["output_root_dir"]

    b = cfg["board"]
    _, dict_id = get_aruco_dict_by_name(b["dict_name"])
    board = {
        "dict_id": dict_id,
        "squares_x": int(b["squares_x"]),
        "squares_y": int(b["squares_y"]),
        "square_len_m": float(b["square_len_mm"]) / 1000.0,
        "marker_len_m": float(b["marker_len_mm"]) / 1000.0,
    }

    c = cfg["camera"]
    camera = {
        "fps": int(c["fps"]),
        "width": int(c["width"]),
        "height": int(c["height"]),
        "exposure_settle_sec": float(c["exposure_settle_sec"]),
        "min_corners_to_save": int(c["min_corners_to_save"]),
        "use_refinement": bool(c["use_refinement"]),
        "series_name": str(c["series_name"]),
    }

    return {"output_root_dir": out_root, "board": board, "camera": camera}


# -------------------------------
# ChArUco helpers
# -------------------------------
def build_charuco_board(dict_id: int, squares_x: int, squares_y: int,
                        square_len_m: float, marker_len_m: float):
    """Build a ChArUco board matching the printed one."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    try:
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_len_m, marker_len_m, aruco_dict
        )
    except Exception:
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_len_m, marker_len_m, aruco_dict
        )
    return aruco_dict, board

def detect_charuco(img_bgr, aruco_dict, board, use_refine=True):
    """Detect ArUco -> refine -> interpolate ChArUco on a BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # New API if available
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    except Exception:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0:
        return 0, None, None, gray

    if use_refine:
        try:
            corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(
                image=gray, board=board, detectedCorners=corners, detectedIds=ids,
                rejectedCorners=rejected, parameters=params
            )
        except Exception:
            pass

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners, markerIds=ids, image=gray, board=board
    )
    n = int(retval) if retval is not None else 0
    return n, charuco_corners, charuco_ids, gray

def estimate_charuco_pose(ch_corners, ch_ids, board, camera_matrix, dist_coeffs=None):
    """Estimate ChArUco board pose in camera frame. Returns (rvec, tvec) or (None, None)."""
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
        rvec=np.zeros((3,1), dtype=np.float32),  # in/out per OpenCV API
        tvec=np.zeros((3,1), dtype=np.float32)
    )
    if retval:
        return rvec, tvec
    return None, None

def draw_overlays(img_bgr, corners, ids, n_charuco, *, idx=None,
                  rvec=None, tvec=None, camera_matrix=None, dist_coeffs=None,
                  axis_len_m=None, min_req=12):
    """Draw HUD, detected corners, and optional pose axes."""
    vis = img_bgr.copy()

    # Detected charuco
    try:
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
    except Exception:
        pass

    # HUD
    cv2.putText(
        vis, f"ChArUco corners: {n_charuco}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 0) if n_charuco >= min_req else (0, 0, 255), 2
    )
    cv2.putText(
        vis, "SPACE: save  |  A: auto-capture  |  Q/ESC: quit",
        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2
    )
    if idx is not None:
        cv2.putText(
            vis, f"Capture Index: {idx}",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2
        )

    # Optional axes
    if rvec is not None and tvec is not None and camera_matrix is not None and dist_coeffs is not None:
        try:
            rvec_cv = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec_cv = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
            D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
            axis_len = float(axis_len_m) if axis_len_m is not None else 0.05
            cv2.drawFrameAxes(vis, K, D, rvec_cv, tvec_cv, axis_len)
        except Exception as e:
            cv2.putText(
                vis, f"[pose draw err] {str(e)}",
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2
            )
    return vis


# -------------------------------
# Main
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="RealSense ChArUco capture.")
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    return ap.parse_args()

def main():
    args = parse_args()
    params = load_config(args.config)

    out_root = os.path.join(params["output_root_dir"], "camera")
    cap_cfg  = params["camera"]
    board_cfg= params["board"]

    # Create output directory: <output_root_dir>/images
    save_dir = os.path.join(out_root, "images")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[Info] Waiting for RealSense frames...")

    # RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, cap_cfg["width"], cap_cfg["height"], rs.format.bgr8, cap_cfg["fps"])
    profile = None

    try:
        profile = pipeline.start(config)
        print(f"[Info] RealSense started {cap_cfg['width']}x{cap_cfg['height']} @ {cap_cfg['fps']} FPS")
        print("[Info] Stabilizing exposure...")
        time.sleep(cap_cfg["exposure_settle_sec"])

        # Dump SDK intrinsics for sanity check
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        sdk_intr_json = {
            "width": intr.width, "height": intr.height,
            "ppx": intr.ppx, "ppy": intr.ppy, "fx": intr.fx, "fy": intr.fy,
            "model": str(intr.model), "coeffs": list(intr.coeffs),
            "K": [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
            "D": list(intr.coeffs),
        }

        with open(os.path.join(out_root, "realsense_rgb_sdk_intrinsics.json"), "w") as f:
            json.dump(sdk_intr_json, f, indent=2)

        # Camera matrix and distortion
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]], dtype=np.float64)
        D = np.array(intr.coeffs, dtype=np.float64).reshape(-1, 1)

        # Build ChArUco board
        aruco_dict, board = build_charuco_board(
            board_cfg["dict_id"],
            board_cfg["squares_x"], board_cfg["squares_y"],
            board_cfg["square_len_m"], board_cfg["marker_len_m"]
        )

        idx = 0
        auto_capture = False
        min_req = cap_cfg["min_corners_to_save"]

        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            frame_bgr = np.asanyarray(color.get_data())

            # Detect + pose
            n, ch_corners, ch_ids, gray = detect_charuco(
                frame_bgr, aruco_dict, board, cap_cfg["use_refinement"]
            )
            rvec, tvec = estimate_charuco_pose(ch_corners, ch_ids, board, K, D)

            vis = draw_overlays(
                frame_bgr, ch_corners, ch_ids, n,
                idx=idx, rvec=rvec, tvec=tvec, camera_matrix=K, dist_coeffs=D,
                axis_len_m=board_cfg["square_len_m"], min_req=min_req
            )
            cv2.imshow("RealSense RGB (ChArUco)", vis)
            key = cv2.waitKey(1) & 0xFF

            # Auto-save when enough corners
            if auto_capture and n >= min_req:
                out_name = f"{cap_cfg['series_name']}_{idx:04d}.png"
                cv2.imwrite(os.path.join(save_dir, out_name), frame_bgr)
                print(f"[Saved] {out_name}  (corners={n})")
                idx += 1
                time.sleep(1)  # debounce

            # Keyboard controls
            if key == ord('a'):
                auto_capture = not auto_capture
                print(f"[Toggle] auto-capture = {auto_capture}")
            elif key == ord(' ') and n >= min_req:
                out_name = f"{cap_cfg['series_name']}_{idx:04d}.png"
                cv2.imwrite(os.path.join(save_dir, out_name), frame_bgr)
                print(f"[Saved] {out_name}  (corners={n})")
                idx += 1
            elif key in (ord('q'), 27):
                break

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print(f"[Done] Images saved to: {save_dir}")
        print(f"[Done] SDK intrinsics saved to: {save_dir}/realsense_rgb_sdk_intrinsics.json")

if __name__ == "__main__":
    main()

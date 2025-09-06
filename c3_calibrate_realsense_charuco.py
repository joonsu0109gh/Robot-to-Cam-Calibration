#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ChArUco calibration + pose+projection comparison on the SAME detections
(RealSense SDK intrinsics vs. Estimated intrinsics).
"""

import os
import cv2
import json
import yaml
import argparse
import numpy as np
import csv
from typing import Generator, Tuple, Optional, List


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
    """Load YAML config and return merged dict with defaults."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    board = cfg.get("board", {})
    dict_name = board.get("dict_name", "DICT_5X5_100")
    aruco_dict, dict_id = get_aruco_dict_by_name(dict_name)

    return {
        "output_root_dir": cfg.get("output_root_dir", "./output"),
        "board": {
            "dict_id": dict_id,
            "aruco_dict": aruco_dict, 
            "dict_name": dict_name,
            "squares_x": int(board.get("squares_x", 5)),
            "squares_y": int(board.get("squares_y", 5)),
            "square_len_m": float(board.get("square_len_mm", 35.0)) / 1000.0,
            "marker_len_m": float(board.get("marker_len_mm", 28.0)) / 1000.0,
        },
    }


# =========================
# ChArUco calibrator
# =========================
class CharucoCalibrator:
    """Base class for ChArUco-based calibration and pose estimation."""

    def __init__(
        self,
        aruco_dict_id: int,
        squares_vertically: int,
        squares_horizontally: int,
        square_length_m: float,
        marker_length_m: float,
        calibration_images_dir: str,
    ):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.squares_vertically = int(squares_vertically)
        self.squares_horizontally = int(squares_horizontally)
        self.square_length = float(square_length_m)
        self.marker_length = float(marker_length_m)
        self.calibration_image_dir = calibration_images_dir

        # Create board
        try:
            self.board = cv2.aruco.CharucoBoard(
                (self.squares_horizontally, self.squares_vertically),
                self.square_length, self.marker_length, self.aruco_dict
            )
        except Exception:
            self.board = cv2.aruco.CharucoBoard_create(
                self.squares_horizontally, self.squares_vertically,
                self.square_length, self.marker_length, self.aruco_dict
            )

        # Detector parameters
        try:
            self.params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
        except Exception:
            self.params = cv2.aruco.DetectorParameters_create()
            self.detector = None  # legacy API

    # -------- I/O helpers --------
    def image_file_list(self) -> List[str]:
        """Return sorted list of image file paths."""
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        paths = [
            os.path.join(self.calibration_image_dir, f)
            for f in sorted(os.listdir(self.calibration_image_dir))
            if f.lower().endswith(exts)
        ]
        return paths

    def image_generator(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Yield (filename, image) from calibration_images_dir in sorted order."""
        for img_path in self.image_file_list():
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Warn] Failed to load: {img_path}")
                continue
            yield os.path.basename(img_path), img

    # -------- detection --------
    def detect_aruco_markers(
        self, image_bgr: np.ndarray, grayscale: bool = True, verbose: bool = False, image_name: str = ""
    ) -> Tuple[Optional[np.ndarray], Optional[list], Optional[list]]:
        """Detect ArUco markers; returns (ids, corners, rejected)."""
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if grayscale else image_bgr

        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(img)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.params)

        if ids is None or len(ids) == 0:
            if verbose:
                print(f"[Info] No ArUco markers in {image_name}")
            return None, None, None

        # Try refinement
        try:
            corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(
                image=img, board=self.board,
                detectedCorners=corners, detectedIds=ids,
                rejectedCorners=rejected, parameters=self.params
            )
        except Exception:
            pass

        if verbose:
            print(f"[Info] {len(ids)} markers in {image_name}")
        return ids, corners, rejected

    def detect_charuco_corners(
        self, image_bgr: np.ndarray, grayscale: bool = True, verbose: bool = False, image_name: str = ""
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect and interpolate ChArUco corners; returns (ch_ids, ch_corners, gray).
        ch_corners shape: (N,1,2) float32
        """
        ids, corners, _ = self.detect_aruco_markers(image_bgr, grayscale=grayscale, verbose=verbose, image_name=image_name)
        if ids is None or len(ids) == 0:
            if verbose:
                print(f"[Info] No ChArUco (no markers) in {image_name}")
            return None, None, None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if grayscale else image_bgr
        retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=self.board
        )

        # Guard against None to avoid cornerSubPix crash
        if retval is None or retval < 1 or ch_corners is None or len(ch_corners) == 0:
            if verbose:
                print(f"[Info] Could not interpolate ChArUco in {image_name}")
            return None, None, gray

        # Sub-pixel refinement (best-effort)
        try:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            cv2.cornerSubPix(gray, ch_corners, (5, 5), (-1, -1), term)
        except Exception:
            pass

        if verbose:
            print(f"[Info] ChArUco corners: {len(ch_corners)} in {image_name}")
        return ch_ids, ch_corners, gray

    # -------- pose --------
    def estimate_pose_charuco(
        self,
        ch_corners: np.ndarray,
        ch_ids: np.ndarray,
        K: np.ndarray,
        D: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate board pose w.r.t camera using provided intrinsics.
        Returns (rvec, tvec) or (None, None).
        """
        if ch_corners is None or ch_ids is None or len(ch_corners) < 4:
            return None, None
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charucoCorners=ch_corners,
            charucoIds=ch_ids,
            board=self.board,
            cameraMatrix=K,
            distCoeffs=D,
            rvec=np.zeros((3,1), dtype=np.float32),
            tvec=np.zeros((3,1), dtype=np.float32),
        )
        if retval:
            return rvec, tvec
        return None, None

    # -------- board points --------
    def get_board_corners_3d(self) -> np.ndarray:
        """
        Return ChArUco chessboard 3D corner coordinates (Nx3) in board frame.
        Supports both old attr and new method APIs.
        """
        pts = None
        if hasattr(self.board, "getChessboardCorners"):
            try:
                pts = self.board.getChessboardCorners()
            except Exception:
                pts = None
        if pts is None and hasattr(self.board, "chessboardCorners"):
            pts = self.board.chessboardCorners
        if pts is None:
            raise RuntimeError("Cannot access board chessboard corners (OpenCV contrib needed).")
        return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


class PinholeCalibrator(CharucoCalibrator):
    """Pinhole calibration using ChArUco."""

    def calibrate(
        self,
        img_size_wh: Tuple[int, int],
        save_json_path: str,
        grayscale: bool = True,
        verbose: bool = True,
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Run cv2.aruco.calibrateCameraCharuco and save K/D to JSON. Returns (ok, K, D)."""
        all_ch_corners = []
        all_ch_ids = []

        for name, img in self.image_generator():
            ch_ids, ch_corners, _ = self.detect_charuco_corners(img, grayscale=grayscale, verbose=verbose, image_name=name)
            if ch_ids is not None and ch_corners is not None and len(ch_corners) >= 4:
                all_ch_corners.append(ch_corners)
                all_ch_ids.append(ch_ids)
            else:
                if verbose:
                    print(f"[Warn] Skipping (insufficient corners): {name}")

        if len(all_ch_corners) == 0:
            print("[Error] No valid ChArUco detections across images.")
            return False, None, None

        retval, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_ch_corners,
            charucoIds=all_ch_ids,
            board=self.board,
            imageSize=img_size_wh,
            cameraMatrix=None,
            distCoeffs=None
        )
        if not retval:
            print("[Error] Calibration failed (retval=0).")
            return False, None, None

        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, "w") as f:
            json.dump({"K": K.tolist(), "D": D.tolist()}, f, indent=2)
        print(f"[OK] Saved intrinsics -> {save_json_path}")
        print("K:\n", K)
        print("D:\n", D.T)
        print(f"[Info] Used {len(all_ch_corners)} images for calibration.")
        return True, K, D


# =========================
# Projection / Drawing
# =========================
def project_points(points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                   K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Project 3D points (Nx3) to image plane using cv2.projectPoints."""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
    img_pts, _ = cv2.projectPoints(points_3d, rvec, tvec, K, D)
    return img_pts.reshape(-1, 2)

def draw_projection_overlay(img_bgr: np.ndarray,
                            pts_rs: Optional[np.ndarray],
                            pts_est: Optional[np.ndarray],
                            label_rs: str = "RS SDK",
                            label_est: str = "Estimated") -> np.ndarray:
    """
    Draw two sets of 2D points on the image with legend + residual lines.
    - RS SDK projections: blue circles
    - Estimated projections: green crosses
    """
    vis = img_bgr.copy()

    # Draw RS points
    if pts_rs is not None:
        for (u, v) in pts_rs.astype(int):
            cv2.circle(vis, (int(u), int(v)), 3, (255, 0, 0), -1)  # blue

    # Draw EST points
    if pts_est is not None:
        for (u, v) in pts_est.astype(int):
            cv2.drawMarker(vis, (int(u), int(v)),
                           color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=8, thickness=1)  # green

    # Residual arrows (RS->EST)
    if pts_rs is not None and pts_est is not None and len(pts_rs) == len(pts_est):
        for (a, b) in zip(pts_rs, pts_est):
            pt1 = tuple(np.round(a).astype(int))
            pt2 = tuple(np.round(b).astype(int))
            cv2.arrowedLine(vis, pt1, pt2, (0, 255, 255), 1, tipLength=0.25)  # yellow

    # Legend
    cv2.rectangle(vis, (10, 10), (350, 90), (0, 0, 0), -1)
    cv2.putText(vis, f"{label_rs}: blue circles", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"{label_est}: green crosses", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Residual: yellow arrows", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,255,200), 1, cv2.LINE_AA)

    return vis

def errors_mae_rmse(pts_a: np.ndarray, pts_b: np.ndarray) -> Tuple[float, float]:
    """Compute MAE and RMSE between two 2D point sets (Nx2)."""
    diff = pts_a - pts_b
    d = np.linalg.norm(diff, axis=1)
    mae = float(np.mean(d))
    rmse = float(np.sqrt(np.mean(d**2)))
    return mae, rmse


# =========================
# CLI / Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="ChArUco pose+projection compare.")
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    ap.add_argument("--num_vis", type=int, default=8,
                    help="Number of images to visualize projection comparison.")
    return ap.parse_args()

def main():
    args = parse_args()
    params = load_config(args.config)

    out_root   = os.path.join(params["output_root_dir"], "camera")
    board_cfg  = params["board"]

    # Fixed folder structure
    IMG_DIR        = os.path.join(out_root, "images")
    SDK_JSON_PATH  = os.path.join(out_root, "realsense_rgb_sdk_intrinsics.json")  # <- keep at root
    VIS_DIR        = os.path.join(out_root, "projection_compare")
    os.makedirs(VIS_DIR, exist_ok=True)

    # Images
    img_files = [f for f in sorted(os.listdir(IMG_DIR))
                 if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tiff"))]
    if not img_files:
        print(f"[Error] No images in: {IMG_DIR}")
        return

    # Get size
    sample_path = os.path.join(IMG_DIR, img_files[0])
    sample_img  = cv2.imread(sample_path)
    if sample_img is None:
        print(f"[Error] Failed to read sample: {sample_path}")
        return
    H, W = sample_img.shape[:2]
    print(f"[Info] Sample image size: {W} x {H}")

    # Build calibrator
    calib = PinholeCalibrator(
        aruco_dict_id=board_cfg["dict_id"],
        squares_vertically=board_cfg["squares_y"],     # vertical = y
        squares_horizontally=board_cfg["squares_x"],   # horizontal = x
        square_length_m=board_cfg["square_len_m"],
        marker_length_m=board_cfg["marker_len_m"],
        calibration_images_dir=IMG_DIR,
    )

    # Calibrate -> K_est, D_est
    out_json_path = os.path.join(out_root, 'estimated_rgb_intrinsics.json')
    ok, K_est, D_est = calib.calibrate(img_size_wh=(W, H), save_json_path=out_json_path, grayscale=True, verbose=True)
    if not ok:
        return

    # Load RealSense SDK intrinsics
    if not os.path.isfile(SDK_JSON_PATH):
        print(f"[Error] Missing RealSense SDK intrinsics JSON: {SDK_JSON_PATH}")
        return
    
    with open(SDK_JSON_PATH, "r") as f:
        sdk = json.load(f)
        
    K_rs = np.array([[sdk["fx"], 0, sdk["ppx"]],
                     [0, sdk["fy"], sdk["ppy"]],
                     [0, 0, 1]], dtype=np.float64)
    D_rs = np.array(sdk.get("coeffs", [0,0,0,0,0]), dtype=np.float64).reshape(-1,1)

    # Prepare board 3D points and helper for selecting observed ones
    board_pts_3d_all = calib.get_board_corners_3d()  # (N,3)
    def select_observed_board_pts(ch_ids: np.ndarray) -> np.ndarray:
        """Return 3D board points for the observed charuco ids."""
        idx = ch_ids.flatten().astype(int)
        return board_pts_3d_all[idx, :]

    # CSV summary
    csv_path = os.path.join(VIS_DIR, "summary.csv")
    csv_rows = [["image", "num_points", "MAE_px", "RMSE_px"]]

    # Visualize first N images
    n_vis = min(args.num_vis, len(img_files))
    for i, name in enumerate(img_files[:n_vis]):
        path = os.path.join(IMG_DIR, name)
        img  = cv2.imread(path)
        if img is None:
            print(f"[Warn] Skip unreadable: {path}")
            continue

        # Detect ChArUco once (shared detections for both poses)
        ch_ids, ch_corners, _ = calib.detect_charuco_corners(img, grayscale=True, verbose=False, image_name=name)
        if ch_ids is None or ch_corners is None or len(ch_corners) < 4:
            print(f"[Warn] Not enough ChArUco corners in {name}; skipping.")
            continue

        # Extract observed 2D measurement array (Nx2)
        ch_2d = ch_corners.reshape(-1, 2)

        # Observed 3D board points for those ids (Nx3)
        pts3d_obs = select_observed_board_pts(ch_ids)

        # Pose with RS K/D
        rvec_rs, tvec_rs = calib.estimate_pose_charuco(ch_corners, ch_ids, K_rs, D_rs)

        # Pose with Estimated K/D
        rvec_est, tvec_est = calib.estimate_pose_charuco(ch_corners, ch_ids, K_est, D_est)

        # Project observed points with each pose+K/D
        pts2d_rs  = project_points(pts3d_obs,  rvec_rs,  tvec_rs,  K_rs,  D_rs)  if rvec_rs  is not None else None
        pts2d_est = project_points(pts3d_obs, rvec_est, tvec_est, K_est, D_est) if rvec_est is not None else None

        # Compute errors (vs. measured ch_2d). We report RS and EST errors separately.
        mae_rs = rmse_rs = mae_est = rmse_est = None
        if pts2d_rs is not None:
            mae_rs, rmse_rs = errors_mae_rmse(pts2d_rs, ch_2d)
        if pts2d_est is not None:
            mae_est, rmse_est = errors_mae_rmse(pts2d_est, ch_2d)

        # Compose overlay using BOTH projected sets (RS blue, EST green)
        vis = draw_projection_overlay(img, pts2d_rs, pts2d_est)

        # Text block with simple metrics
        y0 = 115
        cv2.rectangle(vis, (10, y0-25), (410, y0+45), (0,0,0), -1)
        cv2.putText(vis, f"Points: {len(ch_2d)}", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        if mae_rs is not None:
            cv2.putText(vis, f"RS  MAE/RMSE: {mae_rs:.2f} / {rmse_rs:.2f}px", (20, y0+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        if mae_est is not None:
            cv2.putText(vis, f"EST MAE/RMSE: {mae_est:.2f} / {rmse_est:.2f}px", (20, y0+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        out_path = os.path.join(VIS_DIR, f"proj_compare_{i:02d}_{name}")
        cv2.imwrite(out_path, vis)
        print(f"[Saved] {out_path}")

        # Add to CSV rows (per-image, use EST errors primarily; include RS too)
        csv_rows.append([
            name, len(ch_2d),
            f"{mae_est:.4f}" if mae_est is not None else "",
            f"{rmse_est:.4f}" if rmse_est is not None else "",
        ])

    # Write CSV summary
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"[OK] Summary CSV -> {csv_path}")
    print(f"[Done] Projection comparison images saved to: {VIS_DIR}")


if __name__ == "__main__":
    main()

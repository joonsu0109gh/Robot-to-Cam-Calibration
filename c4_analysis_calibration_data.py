#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze ChArUco capture dataset distribution and report gaps.

"""

import os
import json
import yaml
import cv2
import numpy as np
import argparse
import csv
from typing import Tuple, Optional
from collections import Counter

# ----------------------
# Config / Intrinsics IO
# ----------------------
def get_aruco_dict_by_name(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown aruco dict: {name}")
    did = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(did), did

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    board = cfg.get("board", {})
    dict_name = board.get("dict_name", "DICT_5X5_100")
    aruco_dict, dict_id = get_aruco_dict_by_name(dict_name)

    return {
        "output_root_dir": cfg.get("output_root_dir", "./output"),
        "board": {
            "dict_id": dict_id,
            "aruco_dict": aruco_dict,  # 나중에 detection에 바로 활용 가능
            "dict_name": dict_name,
            "squares_x": int(board.get("squares_x", 5)),
            "squares_y": int(board.get("squares_y", 5)),
            "square_len_m": float(board.get("square_len_mm", 35.0)) / 1000.0,
            "marker_len_m": float(board.get("marker_len_mm", 28.0)) / 1000.0,
        },
    }


def load_intrinsics(out_root: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Preference order:
      1) <out_root>/camera/estimated_rgb_intrinsics.json
    Returns (K, D, source_label).
    If none found, synthesize a weak prior K and zero distortion.
    """
    # 1) estimated
    calib_path = os.path.join(out_root, "estimated_rgb_intrinsics.json")
    if os.path.isfile(calib_path):
        calib_data = json.load(open(calib_path, "r"))
        if "K" in calib_data and "D" in calib_data:
            K = np.array(calib_data["K"], dtype=np.float64)
            D = np.array(calib_data["D"], dtype=np.float64).reshape(-1,1)
            return K, D, f"estimated:{calib_path}"
        else:
            print(f"[Warn] malformed estimated intrinsics JSON: {calib_path}")
            raise ValueError(f"malformed estimated intrinsics JSON: {calib_path}")


# ----------------------
# Board / Detection
# ----------------------
class CharucoHelper:
    def __init__(self, dict_id:int, squares_x:int, squares_y:int, s_len:float, m_len:float):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        try:
            self.board = cv2.aruco.CharucoBoard((squares_x, squares_y), s_len, m_len, self.aruco_dict)
        except Exception:
            self.board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, s_len, m_len, self.aruco_dict)
        try:
            self.params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
        except Exception:
            self.params = cv2.aruco.DetectorParameters_create()
            self.detector = None

    def detect(self, bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, rej = self.detector.detectMarkers(gray)
        else:
            corners, ids, rej = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        if ids is None or len(ids)==0:
            return None, None
        try:
            corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(gray, self.board, corners, ids, rej, parameters=self.params)
        except Exception:
            pass
        retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
        if retval is None or retval < 1 or ch_corners is None:
            return None, None
        # subpixel (best-effort)
        try:
            term = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            cv2.cornerSubPix(gray, ch_corners, (5,5), (-1,-1), term)
        except Exception:
            pass
        return ch_ids, ch_corners  # shapes: (N,1) and (N,1,2)

    def pose(self, ch_corners, ch_ids, K, D) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if ch_corners is None or ch_ids is None or len(ch_corners)<4:
            return None, None
        K = np.asarray(K, dtype=np.float64).reshape(3,3)
        D = np.asarray(D, dtype=np.float64).reshape(-1,1)
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, self.board, K, D,
            rvec=np.zeros((3,1), np.float32), tvec=np.zeros((3,1), np.float32)
        )
        if retval: return rvec, tvec
        return None, None

# ----------------------
# Metrics
# ----------------------
def rodrigues_to_euler(rvec: np.ndarray) -> Tuple[float,float,float]:
    """Return yaw(Z), pitch(Y), roll(X) in degrees using Rz*Ry*Rx convention."""
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        yaw = np.degrees(np.arctan2(R[1,0], R[0,0]))          # around Z
        pitch = np.degrees(np.arctan2(-R[2,0], sy))           # around Y
        roll = np.degrees(np.arctan2(R[2,1], R[2,2]))         # around X
    else:  # gimbal lock
        yaw = np.degrees(np.arctan2(-R[0,1], R[1,1]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        roll = 0.0
    return float(yaw), float(pitch), float(roll)

def convex_hull_area_ratio(pts: np.ndarray, img_wh: Tuple[int,int]) -> float:
    """Convex hull area of 2D points / image area."""
    if pts is None or len(pts)<3: return 0.0
    pts2 = pts.reshape(-1,2).astype(np.float32)
    hull = cv2.convexHull(pts2)
    area = cv2.contourArea(hull)
    W, H = img_wh
    return float(area / (W*H))

def edge_coverage_score(pts: np.ndarray, img_wh: Tuple[int,int], edge_frac: float=0.1) -> float:
    """
    Fraction of points falling in edge bands (left/right/top/bottom each 'edge_frac' wide).
    High value implies good edge evidence (lens distortion observability).
    """
    if pts is None or len(pts)==0: return 0.0
    W, H = img_wh
    x = pts[:,0]; y = pts[:,1]
    xl = x < W*edge_frac
    xr = x > W*(1-edge_frac)
    yt = y < H*edge_frac
    yb = y > H*(1-edge_frac)
    mask = xl | xr | yt | yb
    return float(np.mean(mask))

def brightness_mean(gray: np.ndarray) -> float:
    return float(np.mean(gray))

def sharpness_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def reprojection_error(ch_2d: np.ndarray, pts3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, D: np.ndarray) -> Tuple[float,float]:
    """Return (MAE, RMSE) of projected 3D vs measured 2D."""
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, D)
    proj = proj.reshape(-1,2)
    d = np.linalg.norm(proj - ch_2d, axis=1)
    mae = float(np.mean(d)); rmse = float(np.sqrt(np.mean(d**2)))
    return mae, rmse

def ids_histogram(ch_ids: np.ndarray, total_ids: int) -> np.ndarray:
    """Return count per id index (size=total_ids)."""
    hist = np.zeros(total_ids, dtype=int)
    if ch_ids is None: return hist
    idx = ch_ids.flatten().astype(int)
    idx = idx[(idx>=0) & (idx<total_ids)]
    cnt = Counter(idx)
    for k,v in cnt.items(): hist[k] = v
    return hist

def grid_heatmap_counts(pts: np.ndarray, img_wh: Tuple[int,int], gx:int=6, gy:int=6) -> np.ndarray:
    """Count points per grid cell (gy x gx)."""
    W, H = img_wh
    if pts is None or len(pts)==0: return np.zeros((gy,gx), dtype=int)
    xs = np.clip((pts[:,0]/W*gx).astype(int), 0, gx-1)
    ys = np.clip((pts[:,1]/H*gy).astype(int), 0, gy-1)
    M = np.zeros((gy,gx), dtype=int)
    for x,y in zip(xs,ys): M[y,x]+=1
    return M

# ----------------------
# Plot helpers (no seaborn)
# ----------------------
def save_histograms(out_dir: str, yaw_list, pitch_list, dist_list, bright_list, sharp_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pose
    plt.figure()
    plt.hist(yaw_list, bins=24)
    plt.title("Yaw distribution (deg)")
    plt.xlabel("deg"); plt.ylabel("count")
    plt.savefig(os.path.join(out_dir, "pose_hist_yaw.png")); plt.close()

    plt.figure()
    plt.hist(pitch_list, bins=24)
    plt.title("Pitch distribution (deg)")
    plt.xlabel("deg"); plt.ylabel("count")
    plt.savefig(os.path.join(out_dir, "pose_hist_pitch.png")); plt.close()

    plt.figure()
    plt.hist(dist_list, bins=24)
    plt.title("Distance distribution (m)")
    plt.xlabel("m"); plt.ylabel("count")
    plt.savefig(os.path.join(out_dir, "pose_hist_distance.png")); plt.close()

    # Brightness & Sharpness
    plt.figure()
    plt.hist(bright_list, bins=24)
    plt.title("Brightness mean (0-255)")
    plt.xlabel("mean gray"); plt.ylabel("count")
    plt.savefig(os.path.join(out_dir, "brightness_hist.png")); plt.close()

    plt.figure()
    plt.hist(sharp_list, bins=24)
    plt.title("Sharpness (Laplacian var)")
    plt.xlabel("variance"); plt.ylabel("count")
    plt.savefig(os.path.join(out_dir, "sharpness_hist.png")); plt.close()

def save_heatmap(out_path: str, M: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(M, interpolation="nearest")
    plt.title("Corner density heatmap (gy x gx)")
    plt.colorbar()
    plt.savefig(out_path); plt.close()

# ----------------------
# Main
# ----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Analyze ChArUco dataset coverage and gaps.")
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    ap.add_argument("--grid", type=str, default="6x6", help="Grid for heatmap, e.g., 6x6 or 8x5")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_root = os.path.join(cfg["output_root_dir"], "camera")
    b = cfg["board"]

    img_dir = os.path.join(out_root, "images")
    ana_dir = os.path.join(out_root, "analysis")
    os.makedirs(ana_dir, exist_ok=True)

    # Gather images
    imgs = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tiff"))]
    if not imgs:
        print(f"[Error] no images in {img_dir}")
        return

    # Image size from first
    s0 = cv2.imread(os.path.join(img_dir, imgs[0]))
    if s0 is None:
        print("[Error] cannot read first image")
        return
    H, W = s0.shape[:2]

    # Intrinsics
    K, D, src = load_intrinsics(out_root)
    print(f"[Info] Intrinsics source: {src}")

    # Board / detector
    helper = CharucoHelper(b["dict_id"], b["squares_x"], b["squares_y"], b["square_len_m"], b["marker_len_m"])
    total_board_corners = b["squares_x"]*b["squares_y"]

    # Accumulators
    per_frame = []
    id_hist = np.zeros(total_board_corners, dtype=int)
    yaw_l=[]; pitch_l=[]; dist_l=[]; bright_l=[]; sharp_l=[]
    heatmap_grid = tuple(int(x) for x in args.grid.lower().split("x"))
    gy, gx = heatmap_grid[0], heatmap_grid[1]
    M = np.zeros((gy,gx), dtype=int)

    for name in imgs:
        path = os.path.join(img_dir, name)
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        ch_ids, ch_corners = helper.detect(bgr)
        n = int(ch_corners.shape[0]) if ch_corners is not None else 0
        ch2d = ch_corners.reshape(-1,2) if n>0 else None

        # basic quality metrics
        bright = brightness_mean(gray)
        sharp = sharpness_laplacian(gray)

        # pose + errors (if detectable)
        yaw=pitch=roll=dist=None
        mae=rmse=None
        if n >= 4:
            # pose
            rvec, tvec = helper.pose(ch_corners, ch_ids, K, D)
            if rvec is not None:
                yaw, pitch, roll = rodrigues_to_euler(rvec)
                dist = float(np.linalg.norm(tvec))
                # reprojection error vs measured 2D (observed ids only)
                # Need 3D points for those ids: CharucoBoard has chessboard corners in board frame
                pts3d_all = None
                if hasattr(helper.board, "getChessboardCorners"):
                    pts3d_all = helper.board.getChessboardCorners()
                elif hasattr(helper.board, "chessboardCorners"):
                    pts3d_all = helper.board.chessboardCorners
                pts3d_all = np.asarray(pts3d_all, dtype=np.float64).reshape(-1,3)
                idx = ch_ids.flatten().astype(int)
                pts3d_obs = pts3d_all[idx,:]
                mae, rmse = reprojection_error(ch2d, pts3d_obs, rvec, tvec, K, D)

        # coverage metrics
        hull_ratio = convex_hull_area_ratio(ch2d, (W,H)) if n>0 else 0.0
        edge_cov = edge_coverage_score(ch2d, (W,H), edge_frac=0.1) if n>0 else 0.0

        # id coverage
        if ch_ids is not None:
            idx = ch_ids.flatten().astype(int)
            for k,v in Counter(idx).items():
                if 0 <= k < total_board_corners:
                    id_hist[k] += v

        # heatmap
        if ch2d is not None:
            M += grid_heatmap_counts(ch2d, (W,H), gx=gx, gy=gy)

        per_frame.append(dict(
            image=name,
            corners=n,
            hull_area_ratio=hull_ratio,
            edge_coverage=edge_cov,
            brightness=bright,
            sharpness=sharp,
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, distance_m=dist,
            reproj_mae_px=mae, reproj_rmse_px=rmse
        ))

        # for hists (only if valid)
        if yaw is not None:
            yaw_l.append(yaw); pitch_l.append(pitch); dist_l.append(dist)
        bright_l.append(bright); sharp_l.append(sharp)

    # Save per-frame CSV
    csv_path = os.path.join(ana_dir, "per_frame_metrics.csv")
    keys = list(per_frame[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per_frame)
    print(f"[OK] per-frame csv -> {csv_path}")

    # Save ID coverage
    id_csv = os.path.join(ana_dir, "id_coverage.csv")
    with open(id_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["charuco_id","count"])
        for i,c in enumerate(id_hist):
            w.writerow([i,int(c)])
    print(f"[OK] id coverage -> {id_csv}")

    # Save plots
    save_histograms(ana_dir, yaw_l, pitch_l, dist_l, bright_l, sharp_l)
    save_heatmap(os.path.join(ana_dir, "corner_heatmap.png"), M)

    # ---------- Gap analysis (simple rules) ----------
    gaps = []

    # 1) Pose diversity: yaw/pitch coverage
    def coverage_gap(vals, name, bins, min_per_bin):
        if not vals: return
        hist, edges = np.histogram(vals, bins=bins)
        poor = [f"{edges[i]:.0f}~{edges[i+1]:.0f}" for i,c in enumerate(hist) if c < min_per_bin]
        if poor:
            gaps.append({
                "type": f"{name}_bins_underfilled",
                "bins": [f"{edges[i]:.0f}~{edges[i+1]:.0f}" for i in range(len(edges)-1)],
                "counts": hist.tolist(),
                "underfilled_ranges": poor,
                "recommendation": f"Collect at least {min_per_bin} frames in missing {name} bins."
            })

    coverage_gap(yaw_l,   "yaw_deg",   bins=[-45,-30,-20,-10,0,10,20,30,45], min_per_bin=10)
    coverage_gap(pitch_l, "pitch_deg", bins=[-35,-20,-10,0,10,20,35],       min_per_bin=10)

    # 2) Distance coverage
    if dist_l:
        q = np.quantile(dist_l, [0.1, 0.5, 0.9]).tolist()
        gaps.append({"type":"distance_quantiles_m", "q10_q50_q90": q})
        # heuristic: want broad spread between q10 and q90
        if (q[2]-q[0]) < 0.25*max(0.5, q[1]):  # not wide enough
            gaps.append({
                "type":"distance_span_low",
                "span": q[2]-q[0],
                "recommendation":"Capture at both near and far ranges to widen baseline (±30% of current median)."
            })

    # 3) Edge coverage
    mean_edge = float(np.nanmean([r["edge_coverage"] for r in per_frame]))
    if mean_edge < 0.25:
        gaps.append({
            "type":"edge_coverage_low",
            "mean_edge_coverage": mean_edge,
            "recommendation":"Place board near image borders and corners more often to excite distortion terms."
        })

    # 4) FoV coverage
    mean_hull = float(np.nanmean([r["hull_area_ratio"] for r in per_frame]))
    if mean_hull < 0.15:
        gaps.append({
            "type":"fov_coverage_low",
            "mean_hull_area_ratio": mean_hull,
            "recommendation":"Increase perspective/scale variety; include oblique angles and different distances."
        })

    # 5) Quality (brightness / sharpness)
    br_vals = np.array([r["brightness"] for r in per_frame], dtype=float)
    sh_vals = np.array([r["sharpness"] for r in per_frame], dtype=float)
    br_lo = float(np.quantile(br_vals, 0.1))
    sh_lo = float(np.quantile(sh_vals, 0.1))
    num_dark = int((br_vals < br_lo).sum())
    num_blur = int((sh_vals < sh_lo).sum())
    if num_dark > 0:
        gaps.append({
            "type":"dark_frames",
            "threshold_mean_gray": br_lo,
            "count": num_dark,
            "recommendation":"Add brighter scenes or increase exposure/lighting for the darkest 10% frames."
        })
    if num_blur > 0:
        gaps.append({
            "type":"blurry_frames",
            "threshold_laplacian_var": sh_lo,
            "count": num_blur,
            "recommendation":"Reduce motion/defocus; use shorter exposure or hold camera steadier."
        })

    # 6) ID coverage (which board areas are unseen)
    unseen_ids = np.where(id_hist==0)[0].tolist()
    if unseen_ids:
        gaps.append({
            "type":"charuco_ids_unseen",
            "count": len(unseen_ids),
            "sample_ids": unseen_ids[:20],
            "recommendation":"Pose the board to expose all regions; move to cover currently unseen squares."
        })

    # 7) Reprojection error health
    rp = [r["reproj_rmse_px"] for r in per_frame if r["reproj_rmse_px"] is not None]
    if rp:
        rp_mean = float(np.mean(rp)); rp_p90 = float(np.quantile(rp, 0.9))
        gaps.append({"type":"reprojection_error_stats", "mean_rmse_px": rp_mean, "p90_rmse_px": rp_p90})
        if rp_p90 > 1.5:  # heuristic threshold
            gaps.append({
                "type":"high_reproj_error_tail",
                "p90_rmse_px": rp_p90,
                "recommendation":"Discard worst frames or re-capture with better corner detection and wider coverage."
            })

    # Save summary
    summary = dict(
        num_images=len(imgs),
        image_size=[W,H],
        intrinsics_source=src,
        metrics=dict(
            mean_edge_coverage=mean_edge,
            mean_hull_area_ratio=mean_hull,
            brightness_q10=br_lo,
            sharpness_q10=sh_lo,
        ),
        gaps=gaps
    )
    with open(os.path.join(ana_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] summary -> {os.path.join(ana_dir, 'summary.json')}")
    print(f"[DONE] Analysis artifacts saved in: {ana_dir}")

if __name__ == "__main__":
    main()

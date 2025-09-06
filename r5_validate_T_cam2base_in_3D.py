#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate T_cam2base calibration in 3D using Plotly.

Outputs (saved to <output_root_dir>/robot/3D_vis/):
  1) ee_in_base.html
     - EE poses in BASE  + BASE axes
  2) board_in_cam.html
     - Board(ChArUco) poses in CAMERA + CAM axes
  3) ee_vs_board_in_base.html
     - EE in BASE  + Board->BASE (transparent)
     - Gray lines connect each EE origin to corresponding Board origin
     - BASE axes
  4) ee_vs_board_aligned_in_base.html
     - EE in BASE  + Board->BASE->EE (aligned)
     - Gray lines connect each EE origin to corresponding aligned Board origin
     - BASE axes
"""

import os
import glob
import json
import argparse
import numpy as np
import yaml
import cv2
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go


# -------------------------------
# Config & IO
# -------------------------------
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    out_root = cfg.get("output_root_dir", "./output")
    return {"output_root_dir": out_root}

def load_detections(det_dir: str):
    """Read detections/*.json -> ee_positions, ee_quats(xyzw), board_rvec, board_tvec."""
    jfiles = sorted(glob.glob(os.path.join(det_dir, "*.json")))
    ee_pos, ee_quat = [], []
    bd_rvec, bd_tvec = [], []
    for jf in jfiles:
        with open(jf, "r") as f:
            j = json.load(f)

        # EE pose (xyzw expected)
        if ("ee_position" in j) and ("ee_orientation" in j):
            p = j["ee_position"]; q = j["ee_orientation"]
        else:
            ee = j.get("ee_pose", {})
            p = ee.get("position_m"); q = ee.get("quat_xyzw")

        ch = j.get("charuco_pose", {})
        rvec = ch.get("rvec", [])
        tvec = ch.get("tvec", [])

        if (p is None) or (q is None) or (len(rvec) != 3) or (len(tvec) != 3):
            continue

        ee_pos.append(p)
        ee_quat.append(q)
        bd_rvec.append(rvec)
        bd_tvec.append(tvec)

    return np.asarray(ee_pos, float), np.asarray(ee_quat, float), np.asarray(bd_rvec, float), np.asarray(bd_tvec, float)

def load_cam2base(robot_dir: str):
    path = os.path.join(robot_dir, "cam2base_calibration.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing cam2base_calibration.json at {path}")
    with open(path, "r") as f:
        j = json.load(f)
    T_cam2base = np.asarray(j["T_cam2base"], dtype=np.float64).reshape(4,4)
    # T_ee2board may or may not exist; keep for completeness (not used in these plots)
    T_ee2board = np.asarray(j.get("T_ee2board", np.eye(4)), dtype=np.float64).reshape(4,4)
    return T_cam2base, T_ee2board


# -------------------------------
# Geometry helpers
# -------------------------------
def quat_to_R(q_xyzw):
    return R.from_quat(np.asarray(q_xyzw, float)).as_matrix()

def rvec_to_R(rvec):
    Rm, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3,1))
    return Rm

def make_T(Rm, t3):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = Rm
    T[:3, 3] = np.asarray(t3, float).reshape(3)
    return T

def transform_board_to_ee_in_base(board_pos_base, board_Rm_base, T_ee2board):
    
    T_ee2base_list = []

    for p, Rm in zip(board_pos_base, board_Rm_base):
        p = np.asarray(p, dtype=np.float64).reshape(3)
        Rm = np.asarray(Rm, dtype=np.float64).reshape(3,3)
        T_board2base_i = np.eye(4, dtype=np.float64)
        T_board2base_i[:3, :3] = Rm
        T_board2base_i[:3, 3] = p

        T_ee2base_i =  T_board2base_i @ T_ee2board

        T_ee2base_list.append(T_ee2base_i)

    board_orientations_ee = [T[:3, :3].tolist() for T in T_ee2base_list]
    board_locations_ee = [T[:3, 3].tolist() for T in T_ee2base_list]

    return board_locations_ee, board_orientations_ee

# -------------------------------
# Plotly drawing helpers
# -------------------------------
def axis_traces(origin, Rm, axis_len=0.06, name_prefix="POSE", line_width=5, opacity=1.0, show_legend=True):
    o = np.asarray(origin, float).reshape(3)
    ends = (o[:,None] + Rm @ (axis_len*np.eye(3))).T  # (3,3)
    colors = ["red", "green", "blue"]
    names  = [f"{name_prefix}-X", f"{name_prefix}-Y", f"{name_prefix}-Z"]
    trs = []
    for i in range(3):
        trs.append(go.Scatter3d(
            x=[o[0], ends[i,0]], y=[o[1], ends[i,1]], z=[o[2], ends[i,2]],
            mode="lines",
            line=dict(width=line_width, color=colors[i]),
            name=names[i], opacity=opacity,
            showlegend=(show_legend and i==0),
            hoverinfo="text",
            text=[f"{names[i]} start", f"{names[i]} end"]
        ))
    return trs, [o] + [ends[i] for i in range(3)]

def origin_marker(origin, name="origin", size=3, symbol="circle", show_legend=False):
    o = np.asarray(origin, float).reshape(3)
    return go.Scatter3d(
        x=[o[0]], y=[o[1]], z=[o[2]],
        mode="markers",
        marker=dict(size=size, symbol=symbol),
        name=name, showlegend=show_legend,
        hoverinfo="text",
        text=[f"{name}: ({o[0]:.3f},{o[1]:.3f},{o[2]:.3f})"]
    )

def compute_ranges(points, margin_ratio=0.1):
    pts = np.asarray(points, float)
    if pts.size == 0:
        return [-0.1,0.1], [-0.1,0.1], [-0.1,0.1]
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    span = np.maximum(mx-mn, 1e-6)
    L = float(span.max())
    c = (mn+mx)/2
    m = margin_ratio*L
    return [c[0]-L/2-m, c[0]+L/2+m], [c[1]-L/2-m, c[1]+L/2+m], [c[2]-L/2-m, c[2]+L/2+m]

def save_fig(traces, title, out_html, all_points, pos_err_list=None, rot_err_deg_list=None):
    xr, yr, zr = compute_ranges(all_points, 0.1)
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X (m)", range=xr, showspikes=False),
            yaxis=dict(title="Y (m)", range=yr, showspikes=False),
            zaxis=dict(title="Z (m)", range=zr, showspikes=False),
            aspectmode="cube",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig = go.Figure(data=traces, layout=layout)

    # Optional: show average errors if provided
    if (pos_err_list is not None) and (rot_err_deg_list is not None):
        avg_pos_err = np.mean(pos_err_list)
        avg_rot_err = np.mean(rot_err_deg_list)
        err_text = f"Avg position error: {avg_pos_err*1000:.1f} mm<br>Avg rotation error: {avg_rot_err:.2f}°"
        fig.add_annotation(
            text=err_text,
            xref="paper", yref="paper",
            x=0.055, y=1.02,
            showarrow=False,
            font=dict(size=12, color="black"),
            align="left",
        )

    fig.update_layout(scene_dragmode="orbit", hovermode="closest")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[Saved] {out_html}")

def _rot_geodesic_deg(R1, R2):
    """Geodesic distance (in degrees) between two rotation matrices."""
    R_rel = R1.T @ R2
    # Clamp for numerical safety
    cos = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

# -------------------------------
# Main
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Plotly visualization for hand–eye data (HTML only).")
    ap.add_argument("-c", "--config", required=True, type=str, help="Path to config.yaml")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_root   = cfg["output_root_dir"]
    robot_dir  = os.path.join(out_root, "robot")
    det_dir    = os.path.join(robot_dir, "detections")
    vis_dir    = os.path.join(robot_dir, "3D_vis")
    os.makedirs(vis_dir, exist_ok=True)

    # Load data
    ee_pos, ee_quat, bd_rvec, bd_tvec = load_detections(det_dir)
    if len(ee_pos) == 0:
        print(f"[Error] No valid detections under {det_dir}")
        return
    T_cam2base, T_ee2board = load_cam2base(robot_dir)

    origin_line_width = 8
    ee_pose_line_width = 6
    board_pose_line_width = 3
    connection_line_width = 7
    axis_len = 0.06

    # ========= 1) EE pose in BASE =========
    traces = []; pts = []
    for i in range(len(ee_pos)):
        R_ee = quat_to_R(ee_quat[i])
        p_ee = ee_pos[i]
        trs, frame_pts = axis_traces(p_ee, R_ee, axis_len=axis_len, name_prefix=f"EE#{i}", line_width=ee_pose_line_width)
        traces += trs
        traces.append(origin_marker(p_ee, name=f"EE#{i} origin", size=3))
        pts.extend(frame_pts)

    # BASE frame
    base_trs, base_pts = axis_traces([0,0,0], np.eye(3), axis_len=axis_len*2, name_prefix="BASE", line_width=origin_line_width)
    traces += base_trs
    pts.extend(base_pts)

    save_fig(traces, "EE poses in BASE", os.path.join(vis_dir, "ee_in_base.html"), pts)

    # ========= 2) Board pose in CAM =========
    traces = []; pts = []
    for i in range(len(bd_tvec)):
        R_bd_cam = rvec_to_R(bd_rvec[i]); p_bd_cam = bd_tvec[i]
        trs, frame_pts = axis_traces(p_bd_cam, R_bd_cam, axis_len=axis_len*1.3, name_prefix=f"BRD#{i}", line_width=board_pose_line_width, opacity=0.9)
        traces += trs
        traces.append(origin_marker(p_bd_cam, name=f"BRD#{i} origin", size=4, symbol="diamond"))
        pts.extend(frame_pts)
    # CAM frame
    cam_trs, cam_pts = axis_traces([0,0,0], np.eye(3), axis_len=axis_len*2, name_prefix="CAM", line_width=origin_line_width)
    traces += cam_trs
    pts.extend(cam_pts)

    save_fig(traces, "Board poses in CAM", os.path.join(vis_dir, "board_in_cam.html"), pts)

    # ========= 3) EE pose + Board->BASE =========
    # Transform board to BASE: BASE_T_B = BASE_T_CAM @ CAM_T_B
    board_pos_base, board_Rm_base = [], []
    for i in range(len(bd_tvec)):
        T_bd2cam = make_T(rvec_to_R(bd_rvec[i]), bd_tvec[i])
        T_bd2base = T_cam2base @ T_bd2cam
        board_pos_base.append(T_bd2base[:3, 3])
        board_Rm_base.append(T_bd2base[:3, :3])

    traces = []; pts = []
    # EE (solid)
    for i in range(len(ee_pos)):
        R_ee = quat_to_R(ee_quat[i]); p_ee = ee_pos[i]
        trs, frame_pts = axis_traces(p_ee, R_ee, axis_len=axis_len, name_prefix=f"EE#{i}", line_width=ee_pose_line_width, opacity=1.0)
        traces += trs
        traces.append(origin_marker(p_ee, name=f"EE#{i} origin", size=3))
        pts.extend(frame_pts)
    # Board->BASE (transparent)
    for i in range(len(board_pos_base)):
        p_b = board_pos_base[i]; R_b = board_Rm_base[i]
        trs, frame_pts = axis_traces(p_b, R_b, axis_len=axis_len*1.3, name_prefix=f"BRD#{i}", line_width=board_pose_line_width, opacity=0.65, show_legend=(i==0))
        traces += trs
        traces.append(origin_marker(p_b, name=f"BRD#{i} origin", size=4, symbol="diamond"))
        pts.extend(frame_pts)
    # Pair (EE origin ↔ Board origin)
    N = min(len(ee_pos), len(board_pos_base))
    for i in range(N):
        p_ee = np.asarray(ee_pos[i], float).reshape(3)
        p_bd = np.asarray(board_pos_base[i], float).reshape(3)
        traces.append(
            go.Scatter3d(
                x=[p_ee[0], p_bd[0]],
                y=[p_ee[1], p_bd[1]],
                z=[p_ee[2], p_bd[2]],
                mode="lines",
                line=dict(width=connection_line_width, color="gray"),
                name=f"pair{i} diff",
                showlegend=False,
                hoverinfo="text",
                text=[f"pair{i}", f"pair{i}"],
            )
        )
        pts += [p_ee, p_bd]
    # BASE frame
    base_trs, base_pts = axis_traces([0,0,0], np.eye(3), axis_len=axis_len*2, name_prefix="BASE", line_width=origin_line_width)
    traces += base_trs
    pts.extend(base_pts)

    save_fig(traces, "EE (solid) vs Board→BASE (transparent)", os.path.join(vis_dir, "ee_vs_board_in_base.html"), pts)

    # ========= 4) Aligned EE pose + Board->BASE -> EE =========
    Q = np.asarray(ee_pos, float)

    board_pos_aligned, board_Rm_aligned = transform_board_to_ee_in_base(board_pos_base, board_Rm_base, T_ee2board)

    traces = []; pts = []
    # EE
    for i in range(len(Q)):
        R_ee = quat_to_R(ee_quat[i]); p_ee = ee_pos[i]
        trs, frame_pts = axis_traces(p_ee, R_ee, axis_len=axis_len, name_prefix=f"EE#{i}", line_width=ee_pose_line_width)
        traces += trs
        traces.append(origin_marker(p_ee, name=f"EE#{i} origin", size=3))
        pts.extend(frame_pts)

    # Board aligned (longer axes)
    for i in range(len(board_pos_aligned)):
        p_b = board_pos_aligned[i]; R_b = board_Rm_aligned[i]
        trs, frame_pts = axis_traces(p_b, R_b, axis_len=axis_len*1.3, name_prefix=f"BRD#{i}-AL", line_width=board_pose_line_width, opacity=0.7, show_legend=(i==0))
        traces += trs
        traces.append(origin_marker(p_b, name=f"BRD#{i}-AL origin", size=4, symbol="diamond"))
        pts.extend(frame_pts)


    # Pair (EE origin ↔ aligned Board origin)
    pos_err_list = []
    rot_err_deg_list = []

    N = min(len(ee_pos), len(board_pos_aligned))
    for i in range(N):
        p_ee = np.asarray(ee_pos[i], float).reshape(3)
        p_bd = np.asarray(board_pos_aligned[i], float).reshape(3)

        R_ee = quat_to_R(ee_quat[i])
        R_bd = np.asarray(board_Rm_aligned[i], float).reshape(3, 3)

        # position & rotation errors
        pos_err = np.linalg.norm(p_ee - p_bd)
        rot_err_deg = _rot_geodesic_deg(R_ee, R_bd)
        pos_err_list.append(pos_err)
        rot_err_deg_list.append(rot_err_deg)

        traces.append(
            go.Scatter3d(
                x=[p_ee[0], p_bd[0]],
                y=[p_ee[1], p_bd[1]],
                z=[p_ee[2], p_bd[2]],
                mode="lines",
                line=dict(width=connection_line_width, color="gray"),
                name=f"pair{i} aligned",
                showlegend=False,
                hoverinfo="text",
                text=[f"pair{i}", f"pair{i}"],
            )
        )
        pts += [p_ee, p_bd]

    # BASE frame
    base_trs, base_pts = axis_traces([0,0,0], np.eye(3), axis_len=axis_len*2, name_prefix="BASE", line_width=origin_line_width)
    traces += base_trs
    pts.extend(base_pts)

    save_fig(traces, "Aligned: EE (solid) vs Board→BASE (aligned)", os.path.join(vis_dir, "ee_vs_board_aligned_in_base.html"), pts, pos_err_list, rot_err_deg_list)

    print("[Done] All HTML figures saved in:", vis_dir)


if __name__ == "__main__":
    main()

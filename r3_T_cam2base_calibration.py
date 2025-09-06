#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eye-to-hand solve from detections.

"""

import os
import json
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
# Original logic
# -------------------------------
def load_data(data_dir):
    """
    Load robot/base and camera/board pairs from detection JSON files.

    """
    T_base2ee_list = []     # (R, t) of EE in base
    T_cam2board_list = []   # (R, t) of board in camera

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname), "r") as f:
            data = json.load(f)

        # Robot EE pose
        ee_pos = np.array(data["ee_position"], dtype=np.float64).reshape(3)
        ee_quat = np.array(data["ee_orientation"], dtype=np.float64)  # xyzw expected
        if ee_quat.shape[0] == 4:
            ee_rot = R.from_quat(ee_quat).as_matrix()
        else:
            continue

        # Build homogeneous transform T_ee2base
        T_ee2base = np.eye(4)
        T_ee2base[:3, :3] = ee_rot
        T_ee2base[:3, 3] = ee_pos

        # ChArUco pose in camera
        rvec = np.array(data["charuco_pose"]["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(data["charuco_pose"]["tvec"], dtype=np.float64).reshape(3, 1)
        if rvec.size == 0 or tvec.size == 0:
            continue

        R_board2cam, _ = cv2.Rodrigues(rvec)
        t_board2cam = tvec.reshape(3)

        T_board2cam = np.eye(4)
        T_board2cam[:3, :3] = R_board2cam
        T_board2cam[:3, 3] = t_board2cam

        # Append as in original code
        T_base2ee_list.append(np.linalg.inv(T_ee2base))
        T_cam2board_list.append(np.linalg.inv(T_board2cam))

    return T_base2ee_list, T_cam2board_list


def world_handeye_calibration(T_world2cam_list, T_base2gripper_list):
    """Ordinary hand-eye calibration."""
    R_world2cam, t_world2cam = [], []
    R_base2gripper, t_base2gripper = [], []

    for i in range(len(T_world2cam_list)):
        R_world2cam.append(T_world2cam_list[i][:3, :3])
        t_world2cam.append(T_world2cam_list[i][:3, 3])
        R_base2gripper.append(T_base2gripper_list[i][:3, :3])
        t_base2gripper.append(T_base2gripper_list[i][:3, 3])

    R_world2cam = np.array(R_world2cam)
    t_world2cam = np.array(t_world2cam)
    R_base2gripper = np.array(R_base2gripper)
    t_base2gripper = np.array(t_base2gripper)

    R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
        R_world2cam, t_world2cam, R_base2gripper, t_base2gripper
    )

    T_base2world = np.eye(4)
    T_base2world[:3, :3] = R_base2world
    T_base2world[:3, 3:] = t_base2world

    T_gripper2cam = np.eye(4)
    T_gripper2cam[:3, :3] = R_gripper2cam
    T_gripper2cam[:3, 3:] = t_gripper2cam

    return T_base2world, T_gripper2cam



# -------------------------------
# CLI
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="eye-to-hand solve from detections.")
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    # Paths derived from config
    out_root = cfg["output_root_dir"]
    robot_dir = os.path.join(out_root, "robot")
    det_dir = os.path.join(robot_dir, "detections")
    os.makedirs(robot_dir, exist_ok=True)

    # Load pairs
    T_base2ee_list, T_cam2board_list = load_data(det_dir)
    print(f"Loaded {len(T_base2ee_list)} samples")

    if len(T_base2ee_list) > 5:
        T_base2cam, T_ee2board = world_handeye_calibration(T_cam2board_list, T_base2ee_list)
        print("[Result] T_base2cam:\n", T_base2cam)
        print("[Result] T_ee2board:\n", T_ee2board)

        T_cam2base = np.linalg.inv(T_base2cam)

        # Save result alongside robot data
        out_json_path = os.path.join(robot_dir, "cam2base_calibration.json")
        out_json = {
            "T_cam2base": T_cam2base.tolist(),
            "T_ee2board": T_ee2board.tolist()
        }
        with open(out_json_path, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"Saved calibration to {out_json_path}")
        print("[Result] T_cam2base:\n", T_cam2base)
    else:
        print("Not enough samples for calibration.")

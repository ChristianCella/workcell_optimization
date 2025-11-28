#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Relative imports
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(base_dir)
import fonts
from transformations import rotm_to_quaternion, get_homogeneous_matrix, get_cartesian_pose
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability
from constant_parameters import TestIkFlow
params = TestIkFlow()

def main():

    # Path setup 
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    model_path = os.path.join(base_dir, "ur5e_utils_mujoco/bringup_ur5e.xml")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base") # Base of the tool
    tool_tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_tip")
    flange_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_flange")

    # Set robot base (matrix A^w_b)
    _, _, A_w_b = get_homogeneous_matrix(0.2, 0.2, 0.2, 0, 0, 0)
    set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

    # Set the base of the tool with respect to the flange
    _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.15, 0.0, 0, 0, 0)
    set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

    # Fixed transformation 'tool base (t1) => tool tip (t)'
    _, _, A_t1_t = get_homogeneous_matrix(0, 0, 0.14, 0, 0, 0)
    set_body_pose(model, data, tool_tip_body_id, A_t1_t[:3, 3], rotm_to_quaternion(A_t1_t[:3, :3]))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start visualizing IK-flow solutions…")

        # Desired joint configuration
        q = np.array([0.9951, 4.5500, -1.5874, 0.5281, -4.0680, 2.2158])
        data.qpos[:6] = q.tolist()
        mujoco.mj_forward(model, data)
        viewer.sync()

        # Get the forward kinematics at a specified frame
        pos, quat = get_cartesian_pose(flange_body_id, data)
        print(f"FK: pos={np.round(pos, 3)}, quat={np.round(quat, 3)}")

        input("Press Enter to close the viewer…")

if __name__ == "__main__":
    main()

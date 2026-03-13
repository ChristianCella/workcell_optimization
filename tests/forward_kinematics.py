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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from parameters import TestIkFlow
params = TestIkFlow()
from create_scene import create_reference_frames,  merge_robot_and_tool, inject_robot_tool_into_scene, add_instance

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(base_dir)
import fonts
from transformations import rotm_to_quaternion, get_homogeneous_matrix
from mujoco_utils import set_body_pose, get_cartesian_pose
from ikflow_inference import FastIKFlowSolver, solve_ik_fast

def main():

    # Path setup 
    tool_filename = "screwdriver_marco.xml"
    robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    obstacle_name = "table_grip.xml" 
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Create the robot + tool model
    _ = merge_robot_and_tool(tool_filename=tool_filename, base_dir=base_dir, output_robot_tool_filename=robot_and_tool_file_name)
    
    # Add the robot + tool to the scene
    merged_scene_path = inject_robot_tool_into_scene(robot_tool_filename=robot_and_tool_file_name, 
                                                     output_scene_filename=output_scene_filename, 
                                                     base_dir=base_dir)
    
    # Add a piece for screwing
    obstacle_path = os.path.join(base_dir, "ur5e_utils_mujoco/screwing_pieces", obstacle_name)
    add_instance(
        merged_scene_path,
        obstacle_path,
        merged_scene_path,
        mesh_source_dir=os.path.join(base_dir, "ur5e_utils_mujoco/screwing_pieces"),
        mesh_target_dir=os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/assets")
    )

    # Create the reference frames
    temp_xml_name = create_reference_frames(base_dir, "ur5e_utils_mujoco/" + output_scene_filename, 1)
    model_path = os.path.join(base_dir, "ur5e_utils_mujoco", temp_xml_name)

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top") # Base of the tool
    tool_tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    flange_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_frame_visual_only")

    # Set robot base (matrix A^w_b)
    _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0, 0, 0)
    set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

    # Set the base of the tool with respect to the flange
    theta = 0.0
    fixed_radius = 0.0455
    _, _, A_ee_t1 = get_homogeneous_matrix(-0.06, 0.0, 0.0455, 0.0, -90.0, 90.0)
    _, _, A_t1_t2 = get_homogeneous_matrix(0.0, fixed_radius - (fixed_radius * np.cos(np.radians(theta))), -fixed_radius * np.sin(np.radians(theta)), theta, 0.0, 0.0)
    A_ee_t2 = A_ee_t1 @ A_t1_t2
    set_body_pose(model, data, tool_base_body_id, A_ee_t2[:3, 3], rotm_to_quaternion(A_ee_t2[:3, :3]))

    # Fixed transformation 'tool base (t1) => tool tip (t)'
    _, _, A_t2_t = get_homogeneous_matrix(0, -0.195, 0.028, 90.0, 0.0, 0.0)
    A_ee_t = A_ee_t1 @ A_t1_t2 @ A_t2_t
    set_body_pose(model, data, tool_tip_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start visualizing IK-flow solutions…")

        # Desired joint configuration
        #q = np.radians([-87, -111, -117, 49, -276, 233])
        #q = np.zeros(6)
        #q = np.array([-1.6475823561297815, -1.799856802026266, -1.5848294496536255, -1.3570835006288071, 1.6309974193572998, 0.8154301047325134])
        q = np.array([-4.235, -1.564,  2.035,  4.242, -1.571, -5.805])
        data.qpos[:6] = q.tolist()
        mujoco.mj_forward(model, data)
        viewer.sync()

        # Get the forward kinematics at a specified frame
        pos, quat = get_cartesian_pose(tool_tip_body_id, data, "euler")
        print(f"{fonts.green}Cartesian pose: {np.round(pos, 3)}{fonts.reset}")
        print(f"{fonts.green}Cartesian orientation: {np.round(quat, 3)}{fonts.reset}")
        input("Press Enter to close the viewer…")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation as R

#* Directory for scene creation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from create_scene import create_reference_frames,  merge_robot_and_tool, inject_robot_tool_into_scene, add_instance
from config import *
rob_params = rob_par

#* Directory for the utilities
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
import fonts
from transformations import rotm_to_quaternion, get_homogeneous_matrix
from mujoco_utils import set_body_pose, get_cartesian_pose

def main():

    # Path setup 
    robot_folder = rob_folder
    robot_name = rob_name
    tool_filename = tool_name
    robot_and_tool_file_name = f"temp_{robot_to_use}_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Create a single xml (robot + tool)
    _ = merge_robot_and_tool(
        robot_filename =robot_name, 
        robot_folder=robot_folder, 
        tool_filename=tool_filename, 
        base_dir=base_dir, 
        output_robot_tool_filename=robot_and_tool_file_name
        )
    
    # Add the robot + tool to the scene
    _ = inject_robot_tool_into_scene(
        robot_tool_filename=robot_and_tool_file_name, 
        output_scene_filename=output_scene_filename, 
        base_dir=base_dir,
        robot_folder=robot_folder
    )
    
    model_path = os.path.join(base_dir, "ur5e_utils_mujoco", output_scene_filename)

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base") 
    tool_tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_frame_visual_only")
    
    # Set robot base 
    _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 180.0)
    set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

    # Set the tool
    if tool_to_use == "welding_gun":
        _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) # Welding gun
        set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3])) # Update tool base
        _, _, A_t1_t = get_homogeneous_matrix(0.0, -0.083033, 0.31549, 45.0, 0.0, 0.0) # Welding gun
    elif tool_to_use == "screwdriver":
        _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, -45.0) # Screwdriver
        set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3])) # Update tool base
        _, _, A_t1_t = get_homogeneous_matrix(0, -0.195, 0.028, 90.0, 0.0, 0.0) # Screwdriver
    else:
        raise ValueError(f"Unknown tool type: {tool_to_use}")

    # Compute the final tool tip pose
    A_ee_t = A_ee_t1 @ A_t1_t
    set_body_pose(model, data, tool_tip_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3])) # Update tool tip

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to compute forward kinematics…")

        # Desired joint configuration
        q = rob_params.home_configuration
        data.qpos[:rob_params.nu] = q.tolist()
        mujoco.mj_forward(model, data)
        viewer.sync()

        # Get the forward kinematics at a specified frame
        pos, quat = get_cartesian_pose(ee_body_id, data, "euler")
        print(f"{fonts.green}Cartesian pose: {np.round(pos, 3)}{fonts.reset}")
        print(f"{fonts.green}Cartesian orientation: {np.round(quat, 3)}{fonts.reset}")
        input("Press Enter to close the viewer…")

if __name__ == "__main__":
    main()

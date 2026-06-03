#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os

#* Directory for scene creation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from create_scene import merge_robot_and_tool, inject_robot_tool_into_scene, add_instance
from config import *
rob_params = rob_par
from parameters import TestIK
ik_params = TestIK()

#* Directory for the utilities
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
import fonts
from transformations import rotm_to_quaternion, rotm2euler, get_homogeneous_matrix
from mujoco_utils import set_body_pose
from generate_path import create_path
from generate_trajectory import create_trajectory

def main():

    # Path setup 
    robot_folder = rob_folder
    robot_name = rob_name
    tool_filename = tool_name
    robot_and_tool_file_name = f"temp_{robot_to_use}_with_tool.xml"
    piece_name = pie_name
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
    merged_scene_path = inject_robot_tool_into_scene(
        robot_tool_filename=robot_and_tool_file_name, 
        output_scene_filename=output_scene_filename, 
        base_dir=base_dir,
        robot_folder=robot_folder
    )

    # Import the piece with the trajectory
    add_instance(
        base_scene_path=merged_scene_path,
        instance_path=os.path.join(base_dir, "ur5e_utils_mujoco/pieces", piece_name),
        output_path=merged_scene_path,
        mesh_source_dir=os.path.join(base_dir, "ur5e_utils_mujoco/pieces"),
        mesh_target_dir=os.path.join(base_dir, f"ur5e_utils_mujoco/{robot_folder}/assets")
    )
    
    # Add one reference frame
    model_path = os.path.join(base_dir, "ur5e_utils_mujoco", output_scene_filename)

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base")
    tool_tip_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')

    # Set robot base (matrix A^w_b)
    if robot_to_use == "ur5e":
        _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.25, 0.0, 0.0, 180.0) 
    elif robot_to_use == "gofa5":
        _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.25, 0.0, 0.0, 0.0) 
    set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))
    data.qpos[:rob_params.nu] = rob_params.home_configuration.tolist()

    # Set the piece in the environment out of the way (matrix A^w_p)
    if robot_to_use == "ur5e":
        _, _, A_w_p = get_homogeneous_matrix(0.0, -0.75, 0.0, 0.0, 0.0, 0.0) 
    elif robot_to_use == "gofa5":
        _, _, A_w_p = get_homogeneous_matrix(0.75, 0.0, 0.0, 0.0, 0.0, 90.0) 
    set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

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

    #! Fixed matrix A_wl3_ee
    _, _, A_wl3_ee = get_homogeneous_matrix(0.0, 0.1, 0.0, -90.0, 0.0, 0.0)

    # Get the Cartesian path stitched to the piece
    cartesian_frames = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("point_") and name.endswith("_traj"):
            cartesian_frames.append(i)

    cartesian_path = []
    for frame_id in cartesian_frames:
        pos = data.body(frame_id).xpos
        rot = data.body(frame_id).xmat.reshape(3, 3)
        euler_angles = rotm2euler(rot, degrees=False)
        cartesian_path.append((pos, euler_angles))
  
    #! Solve IK on the trajectory
    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start visualizing IK-flow solutions…")
        q_path = []

        if import_data:
            q_path = np.loadtxt(os.path.join(base_dir, f"workcell_optimization/results/q_path_{robot_to_use}_{ik_solver_to_use}.csv"), delimiter=",", skiprows=1)
        else:
            #* Get the path (no trajectory)
            q_path, total_time = create_path(cartesian_path, model, data, tool_tip_site_id, A_w_b, A_ee_t, A_wl3_ee, save_data)
            print(f"{fonts.green}ik optimization completed in {total_time:.2f} seconds!{fonts.reset}")


        #* Time-optimal path parametrization
        q_traj, _, _, _, _ = create_trajectory(
            q_path=q_path,
            rob_params=rob_params,
            dt=1/rob_params.freq,
            solver_wrapper="ecos",
            save_data=save_data,
            robot_to_use=robot_to_use,
            ik_solver_to_use=ik_solver_to_use,
            v_scaling=v_red_per,
            a_scaling=a_red_per
        )
                    
        input("Press Enter to visualize the best path found")
        #* Set robot in the home configuration
        data.qpos[:6] = rob_params.home_configuration.tolist()
        mujoco.mj_forward(model, data)
        viewer.sync()

        #* Updatae the robot configuration along the trajectory
        dt = 1/rob_params.freq
        t0 = time.perf_counter()
        for i, q in enumerate(q_traj):
            data.qpos[:6] = q
            mujoco.mj_forward(model, data)
            viewer.sync()

            target_time = t0 + (i + 1) * dt
            sleep_time = target_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
        input("Press Enter to finish the demo…")

if __name__ == "__main__":
    main()
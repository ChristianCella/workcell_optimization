#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

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
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability
from generate_path import create_path, smooth_q_path
from generate_trajectory import create_trajectory, compute_time_stamps_totg
from densify_path import densify_cartesian_path

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
    piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{piece_to_use}")
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base")
    tool_tip_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')

    # Set robot base (matrix A^w_b)
    if robot_to_use == "ur5e":
        if piece_to_use == "t_shape":
            _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.4, 0.0, 0.0, 180.0) 
            _, _, A_w_p = get_homogeneous_matrix(0.0, -1.0, 0.6, 0.0, 0.0, 180.0)
        elif piece_to_use == "cube":
            _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 180.0)  #0.25 in z
            _, _, A_w_p = get_homogeneous_matrix(0.0, -0.5, 0.0, 0.0, 0.0, 0.0)
        elif piece_to_use == "reconstructed":
            _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.3, 0.0, 0.0, 180.0)
            _, _, A_w_p = get_homogeneous_matrix(0.0, 0.65, -0.2, 0.0, 0.0, -90.0)
        else:
            raise ValueError(f"Unknown piece type: {piece_to_use}")
        _, _, A_wl3_ee = get_homogeneous_matrix(0.0, 0.1, 0.0, -90.0, 0.0, 0.0) #! Fixed
    elif robot_to_use == "gofa5":
        if piece_to_use == "t_shape":
            _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.4, 0.0, 0.0, 0.0)
            _, _, A_w_p = get_homogeneous_matrix(0.0, 1.0, 0.6, 0.0, 0.0, 0.0)
        elif piece_to_use == "cube":
            _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.25, 0.0, 0.0, 0.0) 
            _, _, A_w_p = get_homogeneous_matrix(0.75, 0.0, 0.0, 0.0, 0.0, 90.0)
        elif piece_to_use == "reconstructed":
            _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            _, _, A_w_p = get_homogeneous_matrix(-0.5, 0.0, -0.3, 0.0, 0.0, 0.0)
        else:
            raise ValueError(f"Unknown piece type: {piece_to_use}")
        _, _, A_wl3_ee = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) #! Fixed
    elif robot_to_use == "fanuc_crx_10ia_l":
        if piece_to_use == "t_shape":
            _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            _, _, A_w_p = get_homogeneous_matrix(0.0, 1.0, 0.6, 0.0, 0.0, 0.0)
        elif piece_to_use == "cube":
            _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) 
            _, _, A_w_p = get_homogeneous_matrix(0.75, 0.0, 0.0, 0.0, 0.0, 90.0)
        elif piece_to_use == "reconstructed":
            _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            _, _, A_w_p = get_homogeneous_matrix(-0.5, 0.0, -0.3, 0.0, 0.0, 0.0)
        else:
            raise ValueError(f"Unknown piece type: {piece_to_use}")
        _, _, A_wl3_ee = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) #! Fixed
    else:
        raise ValueError(f"Unknown robot type: {robot_to_use}")
    
    # Set the robot in home
    set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))
    data.qpos[:rob_params.nu] = rob_params.home_configuration.tolist()
    #data.qpos[:rob_params.nu] = np.array([-0.861, 4.956, -1.675, -1.711, 1.57, 3.852]).tolist() 
  
    # Set the piece in the environment
    set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

    # Set the tool
    if tool_to_use == "welding_gun":
        _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) 
        set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3])) 
        _, _, A_t1_t = get_homogeneous_matrix(0.0, -0.083033, 0.31549, 45.0, 0.0, 0.0)
    elif tool_to_use == "screwdriver":
        _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, -45.0) 
        set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3])) 
        _, _, A_t1_t = get_homogeneous_matrix(0, -0.195, 0.028, 90.0, 0.0, 0.0)
    elif tool_to_use == "painting_gun":
        _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) #0.0
        set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3])) 
        _, _, A_t1_t = get_homogeneous_matrix(0.0, 0.0, 0.21, 0.0, 0.0, 0.0) # 0.21
    else:
        raise ValueError(f"Unknown tool type: {tool_to_use}")

    # Compute the final tool tip pose
    A_ee_t = A_ee_t1 @ A_t1_t
    set_body_pose(model, data, tool_tip_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3])) # Update tool tip

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

    #* Densify the Cartesian path (NOTE: use a path homogeneously discretized)
    cartesian_path = densify_cartesian_path(cartesian_path, eef_step=0.005)

    #! Solve IK on the trajectory
    with mujoco.viewer.launch_passive(model, data) as viewer:
        input(f"Press Enter to start visualizing {ik_solver_to_use} solutions…")
        q_path = []

        if import_data:
            q_path = np.loadtxt(os.path.join(base_dir, f"workcell_optimization/results/q_path_{robot_to_use}_{ik_solver_to_use}.csv"), delimiter=",", skiprows=1)
        else:
            #* Get the path (no trajectory)
            q_path, reach, cols, total_time = create_path(cartesian_path, model, data, rob_params, tool_tip_site_id, A_w_b, A_ee_t, A_wl3_ee, save_data)
            q_path = smooth_q_path(q_path, window=20, polyorder=3)  # very light, just a safety net

            if ik_solver_to_use == "dls":
                unreachable = [i for i, v in enumerate(reach) if v == 1]
                print(f"{fonts.green}Waypoints in positions {unreachable} are not reachable{fonts.reset}")

                #* Check a-posteriori possible collisions and manipulability
                for idx, q in enumerate(q_path):
                    data.qpos[:rob_params.nu] = q
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    n_collisions = get_collisions(model, data, verbose=True)
                    manipulability = inverse_manipulability(q, model, data, rob_params, tool_tip_site_id)
                    if n_collisions > 0:
                        print(f"{fonts.red}Waypoint {idx+1} has {n_collisions} collision(s)!{fonts.reset}")
                        input("Press Enter to visualize the collision(s)…")
                        return
                    if manipulability == 1e12:
                        print(f"{fonts.red}Waypoint {idx+1} is in a singular configuration!{fonts.reset}")
                        input("Press Enter to visualize the singularity…")
                        return
            elif ik_solver_to_use == "ikflow": #* Most checks are already built-in
                if sum(cols) > 0:
                    print(f"{fonts.red}Warning: {sum(cols)} waypoint(s) in the path are in collision!{fonts.reset}")
                    return
                if sum(reach) > 0:
                    print(f"{fonts.red}Warning: {sum(reach)} waypoint(s) in the path are unreachable!{fonts.reset}")
                    return
            else:
                raise ValueError(f"Unknown IK solver type: {ik_solver_to_use}")

            #* Display total time
            print(f"{fonts.green}ik optimization completed in {total_time:.2f} seconds!{fonts.reset}")

        #* Time-optimal path parametrization (topp)
        q_traj, _, _, _, _ = create_trajectory(
            q_path=q_path,
            rob_params=rob_params,
            dt=1/rob_params.freq,
            solver_wrapper="ecos",
            #solver_wrapper="seidel",
            #solver_wrapper="cvxpy",
            save_data=save_data,
            robot_to_use=robot_to_use,
            ik_solver_to_use=ik_solver_to_use,
            v_scaling=v_red_per,
            a_scaling=a_red_per
        )
                 
        input("Press Enter to visualize the best path found")
        #* Set robot in the home configuration
        data.qpos[:rob_params.nu] = rob_params.home_configuration.tolist()
        mujoco.mj_forward(model, data)
        viewer.sync()       

        # Wait 2 seconds before starting the motion
        time.sleep(2)

        #* Update the robot configuration along the trajectory
        dt = 1/rob_params.freq
        t0 = time.perf_counter()
        for i, q in enumerate(q_traj):
            data.qpos[:rob_params.nu] = q
            mujoco.mj_forward(model, data)
            viewer.sync()

            target_time = t0 + (i + 1) * dt
            sleep_time = target_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
        input("Press Enter to finish the demo…")

if __name__ == "__main__":
    main()
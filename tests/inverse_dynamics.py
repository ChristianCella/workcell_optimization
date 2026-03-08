#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from parameters import TestIkFlow
params = TestIkFlow()
from create_scene import create_reference_frames,  merge_robot_and_tool, inject_robot_tool_into_scene, add_instance

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(base_dir)
import fonts
from transformations import rotm_to_quaternion, get_homogeneous_matrix
from mujoco_utils import set_body_pose, compute_jacobian
from ikflow_inference import FastIKFlowSolver, solve_ik_fast



def set_joint_configuration(data, model, desired_qpos):
    # Set the position and velocity to 0
    data.qpos[:model.nq] = desired_qpos.copy()
    data.qvel[:model.nq] = 0.0
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.0

def main():

    # Path setup 
    tool_filename = "screwdriver.xml"
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
    #model_path = os.path.join(base_dir, "ur5e_utils_mujoco", "ur5e/ur5e.xml")
    verbose = True

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        #model.opt.timestep = 0.001
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Target poses for the robot
        target_qpos_list = [
            #np.radians([180, -100, 80, -90, -90, -45]),
            #np.radians([90, -120, 70, -70, -100, -50]),
            #np.radians([-75.21, -47.27, 64.57, -120.19, 273.28, -5.69]),
            # np.radians([-90, -90, -90, -90, 90, 0])
            #[1.37382, -1.1993, 1.8893, -2.28067, -1.58667, -0.2213]
            [-1.6475823561297815, -1.799856802026266, -1.5848294496536255, -1.3570835006288071, 1.6309974193572998, 0.8154301047325134]
            #[-2.5074313322650355, -0.7600118678859253, -1.5335054397583008, 1.8785759645649414, 1.6293644905090332, 0.8156852722167969]
        ]

        # Define bodies, geometries and sites
        base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        tool_base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top") # Base of the tool
        tool_tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
        tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')

        # Always use the site's *parent body* for COM and for xfrc_applied
        site_parent_body = model.site_bodyid[tool_tip_site_id]
        parent_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, site_parent_body)
        if verbose: print(f"{fonts.red}The parent body name is: {parent_body_name}{fonts.reset}")

        # External wrench defined in the *world* frame, applied at the tool site
        external_force_world = np.array([0, 0, 0.0]) #! In terms of world coordinates 
        external_torque_world = np.array([0, 0, -20.0])
        site_wrench_world = np.hstack([external_force_world, external_torque_world])  # [Fx,Fy,Fz, Tx,Ty,Tz] at site

        print(f"The current joints are (deg): {np.round(np.degrees(data.qpos[:6]), 2)}")

        with mujoco.viewer.launch_passive(model, data) as viewer:

            # Scan all the poses
            for idx, desired_qpos in enumerate(target_qpos_list):

                input("Press Enter to start the robot configuration...")
                mujoco.mj_resetData(model, data)

                # Set the new robot base (matrix A^w_b)
                _, _, A_w_b = get_homogeneous_matrix(0.5, 0.5, 0, 0, 0, 0)
                set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

                # Set the base of the tool with respect to the flange
                _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0, 0, 0)
                set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

                # Fixed transformation 'tool top (t1) => tool tip (t)'
                _, _, A_t1_t = get_homogeneous_matrix(0, 0, 0.31, 0, 0, 0)
                set_body_pose(model, data, tool_tip_body_id, A_t1_t[:3, 3], rotm_to_quaternion(A_t1_t[:3, :3]))

                # Set the robot in the specified configuration
                set_joint_configuration(data, model, desired_qpos)

                # Simulate for 10 seconds
                seconds_per_config = 10.0
                start_time = time.perf_counter()

                mujoco.mj_forward(model, data)
                viewer.sync()

                input(f"Press enter to enable physics")

                while time.perf_counter() - start_time < seconds_per_config:
                    mujoco.mj_forward(model, data)

                    #! Adjustments required by mujoco
                    p_site = data.site_xpos[tool_tip_site_id]
                    p_com  = data.xipos[site_parent_body]
                    r = p_site - p_com  

                    # Shift torque to COM:  T_com = T_site + r x F
                    F_site = site_wrench_world[:3]
                    T_site = site_wrench_world[3:]
                    T_com  = T_site + np.cross(r, F_site)

                    # Apply the *reaction* wrench at the parent body's COM (world-frame)
                    data.xfrc_applied[site_parent_body, :3] = -F_site #! Applied FROM the environment TO the body (reaction force)
                    data.xfrc_applied[site_parent_body, 3:] = -T_com

                    # Compute torques
                    J6 = compute_jacobian(model, data, tool_tip_site_id)          
                    tau_ext = J6.T @ site_wrench_world  #* +J^T w to counterbalance the reaction
                    gravity_comp = data.qfrc_bias[:6]  #* C(q,qdot) qdot + G(q)
                    data.ctrl[:6] = gravity_comp + tau_ext #* Total torques

                    mujoco.mj_step(model, data)
                    viewer.sync()

                    #* Clear xfrc_applied so we reapply explicitly each loop
                    data.xfrc_applied[site_parent_body, :] = 0.0
                
                # Debugging
                if verbose:
                    print(f"{fonts.green}tau_grav: {np.round(gravity_comp, 2)}{fonts.reset}")
                    print(f"{fonts.cyan}tau_ext (J^T w): {np.round(tau_ext, 2)}{fonts.reset}")
                    print(f"{fonts.yellow}tau_tot: {np.round(data.ctrl[:6], 2)}{fonts.reset}")
                    print(f"{fonts.purple}Real tau: {np.round(data.qfrc_actuator[:6], 2)}{fonts.reset}")

            print("\n--- Finished all configurations. ---")
            input("Press Enter to close the viewer and exit...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

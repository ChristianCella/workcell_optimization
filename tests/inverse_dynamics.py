#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
from scipy.spatial.transform import Rotation as R


# Append the path to 'utils'
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
from transformations import rotm_to_quaternion, get_homogeneous_matrix
from mujoco_utils import set_body_pose, compute_jacobian, scene_manager
import fonts

def set_joint_configuration(data, model, desired_qpos):
    # Set the position and velocity to 0
    data.qpos[:model.nq] = desired_qpos.copy()
    data.qvel[:model.nq] = 0.0
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.0

def main():

    # Path setup 
    ur5e_utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ur5e_utils_mujoco'))
    model_path = scene_manager("robot", 1, ur5e_utils_dir, "bringup_ur5e.xml", "extension.xml")
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
            np.radians([-90, -90, -90, -90, 90, 0])
        ]

        # Define bodies, geometries and sites
        base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        tool_base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base") # Base of the tool
        tool_tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_tip")
        tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_tip_site')

        # Always use the site's *parent body* for COM and for xfrc_applied
        site_parent_body = model.site_bodyid[tool_tip_site_id]
        parent_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, site_parent_body)
        if verbose: print(f"{fonts.red}The parent body name is: {parent_body_name}{fonts.reset}")

        # External wrench defined in the *world* frame, applied at the tool site
        external_force_world = np.array([0, 0, 30]) #! In terms of world coordinates 
        external_torque_world = np.array([0.0, 0, 0])
        site_wrench_world = np.hstack([external_force_world, external_torque_world])  # [Fx,Fy,Fz, Tx,Ty,Tz] at site

        print(f"The current joints are (deg): {np.round(np.degrees(data.qpos[:6]), 2)}")

        with mujoco.viewer.launch_passive(model, data) as viewer:

            # Scan all the poses
            for idx, desired_qpos in enumerate(target_qpos_list):

                input("Press Enter to start the robot configuration...")
                mujoco.mj_resetData(model, data)

                # Set the new robot base (matrix A^w_b)
                _, _, A_w_b = get_homogeneous_matrix(0, 0, 0, 0, 0, 0)
                set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

                # Set the base of the tool with respect to the flange
                _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0, 0, 0)
                set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

                # Fixed transformation 'tool top (t1) => tool tip (t)'
                _, _, A_t1_t = get_homogeneous_matrix(0, 0, 0.32, 0, 0, 0)
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

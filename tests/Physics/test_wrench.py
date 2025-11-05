#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
from scipy.spatial.transform import Rotation as R

# Append the path to 'scene_manager'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scene_manager')))
from create_scene import create_scene

# Append the path to 'utils'
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils'))
sys.path.append(utils_dir)
from transformations import rotm_to_quaternion, get_world_wrench, get_homogeneous_matrix
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, compute_jacobian

def controller(Kp, Kd, Ki, q_desired, q_current, qd_current, error_integral, dt):
    q_error = q_desired - q_current
    qd_error = 0.0 - qd_current
    error_integral += q_error * dt
    control_torque = Kp * q_error + Ki * error_integral + Kd * qd_error
    return control_torque, error_integral

def set_joint_configuration(data, model, desired_qpos):
    # Set the position and velocity to 0
    data.qpos[:7] = desired_qpos.copy()
    data.qvel[:7] = 0.0
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.0

def main():

    # Path setup 
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    model_path = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/ur5e.xml")

    # Variables
    verbose = False           # Decide how verbose the code should be
    robot_motion = False      # Instantaneous placement or motion
    enable_control = False    # Decide whether to use the PID or go open loop
    Kp = np.array([500, 500, 500, 150, 150, 150])
    Kd = np.array([30, 30, 30, 30, 30, 30])
    Ki = np.array([0, 0, 0, 0, 0, 0])

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        #model.opt.timestep = 0.001
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Target poses for the robot
        target_qpos_list = [
            np.radians([180, -100, 80, -90, -90, -45]),
        ]

        # Define bodies, geometries and sites
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
        tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
        screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")

        # Always use the site's *parent body* for COM and for xfrc_applied
        site_parent_body = model.site_bodyid[tool_site_id]

        # External wrench defined in the *world* frame, applied at the tool site
        external_force_world = np.array([0, 0, -30]) #! In terms of world coordinates 
        external_torque_world = np.array([0.0, 0, -30])
        site_wrench_world = np.hstack([external_force_world, external_torque_world])  # [Fx,Fy,Fz, Tx,Ty,Tz] at site

        print(f"The current joints are (deg): {np.round(np.degrees(data.qpos[:6]), 2)}")

        with mujoco.viewer.launch_passive(model, data) as viewer:

            input("Press Enter to start the robot configuration...")

            # Scan all the poses
            for idx, desired_qpos in enumerate(target_qpos_list):

                mujoco.mj_resetData(model, data)

                # Set the new robot base (matrix A^w_b)
                _, _, A_w_b = get_homogeneous_matrix(0, 0, 0, 0, 0, 0)
                set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

                # Set the frame 'screw_top to a new pose wrt flange' and move the screwdriver there
                _, _, A_ee_t1 = get_homogeneous_matrix(0, 0, 0, 0, 0, 0)
                set_body_pose(model, data, screwdriver_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

                # Fixed transformation 'tool top (t1) => tool tip (t)'
                _, _, A_t1_t = get_homogeneous_matrix(0, 0, 0.0, 0, 0, 0)

                # Update the position of the tool tip (Just for visualization purposes)
                A_ee_t = A_ee_t1 @ A_t1_t
                set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

                if not robot_motion:  # You do not want the robot to move, just set the configuration
                    set_joint_configuration(data, model, desired_qpos)
                else:
                    # Enable the Integral action
                    Ki = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

                # Simulate for 10 seconds
                seconds_per_config = 10.0
                start_time = time.perf_counter()
                error_integral = np.zeros(6)

                mujoco.mj_forward(model, data)
                viewer.sync()

                input(f"Press enter to enable physics")

                while time.perf_counter() - start_time < seconds_per_config:
                    mujoco.mj_forward(model, data)

                    # Use the simulation timestep for controller timing
                    dt = model.opt.timestep
                    world_wrench = site_wrench_world.copy()  # already world-frame at the site

                    # === Apply the reaction wrench physically at the site, but written at the parent body COM ===
                    # Site & COM positions (world)
                    p_site = data.site_xpos[tool_site_id]
                    p_com  = data.xipos[site_parent_body]
                    r = p_site - p_com  # vector from COM to site (world)

                    if verbose:
                        print(f"r (COM->site) = {np.round(r, 4)} m")

                    # Shift torque to COM:  T_com = T_site + r x F
                    F_site = world_wrench[:3]
                    T_site = world_wrench[3:]
                    T_com  = T_site + np.cross(r, F_site)

                    # Apply the *reaction* wrench at the parent body's COM (world-frame)
                    data.xfrc_applied[site_parent_body, :3] = -F_site #! Applied on the body (reaction force)
                    data.xfrc_applied[site_parent_body, 3:] = -T_com

                    # === Jacobian at the site (world-frame) ===
                    jacp = np.zeros((3, model.nv))
                    jacr = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id)  # world → site
                    J6 = np.vstack([jacp, jacr])[:, :6]

                    # === Joint torques: gravity/Coriolis + counter torque + optional PD ===
                    tau_ext = J6.T @ world_wrench                # +J^T w to counterbalance the reaction
                    gravity_comp = data.qfrc_bias[:6]            # C(q,qdot) qdot + G(q); at qdot=0 it's just gravity
                    ctrl_torque, error_integral = controller(
                        Kp, Kd, Ki, desired_qpos, data.qpos[:6], data.qvel[:6], error_integral, dt
                    )

                    if enable_control:
                        data.ctrl[:6] = gravity_comp + tau_ext + ctrl_torque
                    else:
                        data.ctrl[:6] = gravity_comp + tau_ext #! If the world wrench is considered

                    # Debugging
                    if verbose:
                        print(f"tau_grav: {np.round(gravity_comp, 2)}")
                        print(f"tau_ext (J^T w): {np.round(tau_ext, 2)}")
                        if enable_control:
                            print(f"tau_PD: {np.round(ctrl_torque, 2)}")
                        print("ctrl:", np.round(data.ctrl[:6], 2))
                        print("qfrc_actuator:", np.round(data.qfrc_actuator[:6], 2))

                    mujoco.mj_step(model, data)
                    viewer.sync()

                    # Keep loop pacing roughly aligned with sim time (optional)
                    time.sleep(model.opt.timestep)

                    # Clear xfrc_applied so we reapply explicitly each loop
                    data.xfrc_applied[site_parent_body, :] = 0.0

                # Final prints
                print(f"tau_gravity: {np.round(gravity_comp, 2)}")
                print(f"tau_ext (J^T w): {np.round(tau_ext, 2)}")
                if enable_control:
                    print(f"Final PD torque (last): {np.round(ctrl_torque, 2)}")
                print("Total commanded torque:", np.round(data.ctrl[:6], 2))
                print("Applied actuator torques:", np.round(data.qfrc_actuator[:6], 2))
                print("Final joint angles (deg):", np.round(np.degrees(data.qpos[:6]), 2))

            print("\n--- Finished all configurations. ---")
            input("Press Enter to close the viewer and exit...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

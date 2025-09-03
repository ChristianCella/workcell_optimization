#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
from scipy.spatial.transform import Rotation as R

''' 
Case 1) robot_motion = False ==> Instantly impose a joint configuration to the robot.
Open-loop torque control + PD compensator to obtain the static equilibrium of a robot under an external wrench.
- The wrench (w) is applied in 'tool_site' and expressed in the world frame.
- The dynamic model of a robot is the following: M(q) * qdd + C(q, qd) * qd + G(q) = tau + J^T * w (qd = qdd = 0 in static equilibrium)
- The total torque tau = G(q) + J^T * w (ideally)
- The Jacobian (J) must be the one from the world (since the force is in world frame) to the point of application of the force
- The PD controller allows to compensate for small numerical errors.

Case 2) robot_motion = True ==> Move the robot to a desired configuration.
- In this case, you do not impose an initial configuration to the robot, but the PID controller will produce the
    torques needed to reach the desired configuration.
- Of course, since the trajectory is not given, the PID struggles in reaching the desired configuration, especially for joint 1.

NOTE: The resolution of the inverse dynamics works perfectly. There is a minor issue:
- if 'model.opt.timestep' is smaller than 0.0001 => no problem
- if 'model.opt.timestep' is larger than 0.0001 => the small numerical errors sum up and the robot diverges
- To avoid having computationally intensive simulations, set 'model.opt.timestep' to 0.002 or leave the default: you can 
    correct the numerical errors by enabling the PD controller.
'''

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
    return control_torque

def set_joint_configuration(data, model, desired_qpos):
    # Set the position and velocity to 0
    data.qpos[:7] = desired_qpos.copy()
    data.qvel[:7] = 0.0
    data.qacc[:7] = 0.0
    mujoco.mj_forward(model, data)
    
def main():

    # Path setup 
    tool_filename = "driller.xml"
    robot_and_tool_file_name = "temp_kuka_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    piece_name1 = "aluminium_plate.xml" 
    piece_name2 = "Linear_guide.xml"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

    # Create the scene

    model_path = create_scene(tool_name=tool_filename, robot_and_tool_file_name=robot_and_tool_file_name,
                              output_scene_filename=output_scene_filename, piece_name1=piece_name1, piece_name2=piece_name2, base_dir=base_dir)

    #model_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/kuka/iiwa14.xml")

    # Variables
    verbose = False #! Decide how verbose the code should be
    robot_motion = False #! Instantaneous placement or motion
    enable_control = False #! Decide whether to use the PID or go open loop
    Kp = np.array([500, 500, 500, 500, 150, 150, 150])
    Kd = np.array([30, 30, 30, 30, 30, 30, 30])
    Ki = np.array([0, 0, 0, 0, 0, 0, 0])

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        model.opt.timestep = 0.00001 
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Target poses for the robot
        target_qpos_list = [
            np.array([1.7382,  0.8907, -2.2000, -1.5925,  0.4470, -1.9824, -1.1051]),
            #np.radians([15.99, -62.07, 121.58, -159.82, -99, -90]),
            #np.radians([-28.83, -84.36, 102.8, 18.69, -98.83, -101.46]),
            #np.radians([-29.77, -160.55, 110.57, -186.16, -98.81, -101.15]),
            #np.radians([-124.35, -158.41, 147, -153.21, -155.31, -98.24])
        ]

        # Define bodies, geometries and sites
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
        tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
        screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")

        # Forces defined in the world frame and applied at the tool center
        external_force_world = np.array([0, 0, -50])
        external_torque_world = np.array([0, 0, 0])
        local_wrench = np.hstack([external_force_world, external_torque_world])

        with mujoco.viewer.launch_passive(model, data) as viewer:

            input("Press Enter to start the robot configuration...")

            # Scan all the poses
            for idx, desired_qpos in enumerate(target_qpos_list):

                mujoco.mj_resetData(model, data)

                print(f"model.nq = {model.nq}, model.nv = {model.nv}")
                for i in range(model.njnt):
                    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    jtype = model.jnt_type[i]
                    jadr = model.jnt_dofadr[i]
                    print(f"Joint {i}: {jname}, type = {jtype}, dof address = {jadr}")


                # Set the new robot base (matrix A^w_b)
                _, _, A_w_b = get_homogeneous_matrix(0.5, 0.5, 0.5, 0, 0, 0)
                set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

                # Set the frame 'screw_top to a new pose wrt flange' and move the screwdriver there
                _, _, A_ee_t1 = get_homogeneous_matrix(0, 0, 0, 0, 0, 0)
                set_body_pose(model, data, screwdriver_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

                # Fixed transformation 'tool top (t1) => tool tip (t)' (NOTE: the rotation around z is not important)
                _, _, A_t1_t = get_homogeneous_matrix(0, 0, 0.26, 0, 0, 0)
                #_, _, A_t1_t = get_homogeneous_matrix(0, 0, 0, 0, 0, 0)

                # Update the position of the tool tip (Just for visualization purposes)
                A_ee_t = A_ee_t1 @ A_t1_t
                set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

                if not robot_motion: # You do not want the robot to move, just set the configuration
                    set_joint_configuration(data, model, desired_qpos)
                else:
                    # Enable the Integral action
                    Ki = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

                # Simulate for 10 seconds
                seconds_per_config = 10.0
                start_time = time.perf_counter()
                last_time = start_time
                error_integral = np.zeros(7)

                while time.perf_counter() - start_time < seconds_per_config:
                    mujoco.mj_forward(model, data)

                    # Compute the integration time
                    current_time = time.perf_counter()
                    dt = current_time - last_time
                    last_time = current_time

                    # Rotate the wrench as the world
                    R_tool_to_world = data.site_xmat[tool_site_id].reshape(3, 3) # Rotation tool => world
                    R_world_to_tool = R_tool_to_world.T # Rotation world => tool
                    world_wrench = get_world_wrench(R_world_to_tool, local_wrench) #! Wrench in world frame

                    # Apply the wrench (NOTE: always expressed in the world frame)
                    data.xfrc_applied[tool_body_id, :3] = -world_wrench[:3] 
                    data.xfrc_applied[tool_body_id, 3:] = -world_wrench[3:]

                    # Get the Jacobian matrix from the world frame to the tool site
                    jacp = np.zeros((3, model.nv))
                    jacr = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id) # From world to tool site
                    J7 = np.vstack([jacp, jacr])[:, :model.nv]

                    # Compute the total torque needed to stabilize the robot                   
                    tau_ext = J7.T @ world_wrench
                    gravity_comp = data.qfrc_bias[:model.nv]
                    control_torque = controller(Kp, Kd, Ki, desired_qpos, data.qpos[:model.nv], data.qvel[:model.nv], error_integral, dt)

                    if enable_control:
                        data.ctrl[:model.nv] = gravity_comp + tau_ext + control_torque # ! Apply the control torque
                    else:
                        data.ctrl[:model.nv] = gravity_comp + tau_ext

                    # Debugging
                    if verbose:
                        print(f"The torque due to gravity is: {np.round(gravity_comp, 2)}")
                        print(f"The torque due to the external action is: {np.round(tau_ext, 2)}")
                        #print(f"The PD torque is: {np.round(control_torque, 2)}")
                        print("Total torque applied to the joints:", np.round(data.ctrl[:model.nv], 2))
                        print("Actual applied torques:", data.qfrc_actuator[:model.nv])


                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)
                    data.xfrc_applied[tool_body_id, :] = 0.0

                # Final prints
                print(f"The torque due to gravity is: {np.round(gravity_comp, 2)}")
                print(f"The torque due to the external action is: {np.round(tau_ext, 2)}")
                #print(f"The PD torque is: {np.round(control_torque, 2)}")
                print("Total torque applied to the joints:", np.round(data.ctrl[:model.nv], 2))
                print("Actual applied torques:", data.qfrc_actuator[:model.nv])
                print("Final joint angles (deg):", np.round(np.degrees(data.qpos[:model.nv]), 2))

            print("\n--- Finished all configurations. ---")
            input("Press Enter to close the viewer and exit...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

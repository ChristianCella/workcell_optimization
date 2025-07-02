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
'''

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def controller(Kp, Kd, Ki, q_desired, q_current, qd_current, error_integral, dt):
    q_error = q_desired - q_current
    qd_error = 0.0 - qd_current
    error_integral += q_error * dt
    control_torque = Kp * q_error + Ki * error_integral + Kd * qd_error
    return control_torque

def set_joint_configuration(data, model, desired_qpos):
    # Set the position and velocity to 0
    data.qpos[:6] = desired_qpos.copy()
    data.qvel[:6] = 0.0
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.0


def main():

    # Path setup
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, "universal_robots_ur5e/scene.xml")

    # Variables
    verbose = False
    robot_motion = True
    enable_control = True
    Kp = np.array([500, 500, 500, 150, 150, 150])
    Kd = np.array([30, 30, 30, 30, 30, 30])
    Ki = np.array([0, 0, 0, 0, 0, 0])

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Target poses for the robot
        target_qpos_list = [
            np.radians([100, -94.96, 101.82, -95.72, -96.35, -40.97]),
            np.radians([15.99, -62.07, 121.58, -159.82, -99, -90]),
            np.radians([-28.83, -84.36, 102.8, 18.69, -98.83, -101.46]),
            np.radians([-29.77, -160.55, 110.57, -186.16, -98.81, -101.15]),
            np.radians([-124.35, -158.41, 147, -153.21, -155.31, -98.24])
        ]

        # Define bodies, geometries and sites
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
        tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")

        # Forces defined in the world frame and applied at the tool center
        external_force_world = np.array([0.0, 5.0, 0.0])
        external_torque_world = np.array([0.0, 0.0, 0.0])
        wrench_world = np.hstack([external_force_world, external_torque_world])

        with mujoco.viewer.launch_passive(model, data) as viewer:

            input("Press Enter to start the robot configuration...")

            # Scan all the poses
            for idx, desired_qpos in enumerate(target_qpos_list):

                # Set the robot base to a new pose
                model.body_pos[base_body_id] = [idx * 0.3, idx * 0.1, 0.5]
                model.body_quat[base_body_id] = euler_to_quaternion(45, 45, 0, degrees=True)

                # Set the tool to a new pose with respect to the ee (flange)
                model.body_pos[tool_body_id] = [0.1, 0.1, 0.1]
                model.body_quat[tool_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)

                if not robot_motion: # You do not want the robot to move, just set the configuration
                    set_joint_configuration(data, model, desired_qpos)
                else:
                    # Enable the Integral action
                    Ki = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

                # Simulate for 10 seconds
                seconds_per_config = 10.0
                start_time = time.perf_counter()
                last_time = start_time
                error_integral = np.zeros(6)

                while time.perf_counter() - start_time < seconds_per_config:
                    mujoco.mj_forward(model, data)

                    # Compute the integration time
                    current_time = time.perf_counter()
                    dt = current_time - last_time
                    last_time = current_time

                    # Apply the wrench in the simulation (of course, opposite to the previous)
                    data.xfrc_applied[tool_body_id, :3] = -external_force_world
                    data.xfrc_applied[tool_body_id, 3:] = -external_torque_world

                    # Get the Jacobians
                    jacp = np.zeros((3, model.nv))
                    jacr = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id) # From world to tool site
                    J6 = np.vstack([jacp, jacr])[:, :6]

                    # Compute the total torque needed to stabilize the robot                   
                    tau_ext = J6.T @ wrench_world
                    gravity_comp = data.qfrc_bias[:6]
                    control_torque = controller(Kp, Kd, Ki, desired_qpos, data.qpos[:6], data.qvel[:6], error_integral, dt)

                    if enable_control:
                        data.ctrl[:6] = gravity_comp + tau_ext + control_torque # ! Apply the control torque
                    else:
                        data.ctrl[:6] = gravity_comp + tau_ext

                    # Debugging
                    if verbose:
                        print(f"The torque due to gravity is: {np.round(gravity_comp, 2)}")
                        print(f"The torque due to the external action is: {np.round(tau_ext, 2)}")
                        print(f"The PD torque is: {np.round(control_torque, 2)}")
                        print("Total torque applied to the joints:", np.round(data.ctrl[:6], 2))
                        print("Actual applied torques:", data.qfrc_actuator[:6])


                    # ? Visualize the wrench components as cylinders

                    # Get geom IDs (do this once, outside the sim loop)
                    force_arrow_geom_ids = [
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "force_arrow_x_geom"),
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "force_arrow_y_geom"),
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "force_arrow_z_geom"),
                    ]
                    moment_arrow_geom_ids = [
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moment_arrow_x_geom"),
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moment_arrow_y_geom"),
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moment_arrow_z_geom"),
                    ]

                    # Visualization parameters
                    force_arrow_length_scale = 0.0035  # meters per Newton
                    force_arrow_base = 0.15
                    moment_arrow_length_scale = 0.0011 # meters per Nm
                    moment_arrow_base = 0.1

                    start = data.site_xpos[tool_site_id]
                    z_axis = np.array([0, 0, 1])

                    # Forces (XYZ)
                    for i in range(3):
                        val = external_force_world[i]
                        direction = np.zeros(3)
                        direction[i] = 1
                        length = force_arrow_base + force_arrow_length_scale * np.abs(val)
                        end = start + np.sign(val) * direction * length

                        # Set mocap body position/orientation for this arrow
                        data.mocap_pos[i, :] = 0.5 * (start + end)
                        if np.linalg.norm(direction) > 1e-8:
                            rot, _ = R.align_vectors([direction], [z_axis])
                            quat = rot.as_quat()
                            data.mocap_quat[i, :] = [quat[3], quat[0], quat[1], quat[2]]
                        else:
                            data.mocap_quat[i, :] = [1, 0, 0, 0]
                        model.geom_size[force_arrow_geom_ids[i]][1] = length / 2  # half-length

                    # Moments (XYZ)
                    for i in range(3):
                        val = external_torque_world[i]
                        direction = np.zeros(3)
                        direction[i] = 1
                        length = moment_arrow_base + moment_arrow_length_scale * np.abs(val)
                        end = start + np.sign(val) * direction * length

                        idx = i + 3  # mocap/body index for moments
                        data.mocap_pos[idx, :] = 0.5 * (start + end)
                        if np.linalg.norm(direction) > 1e-8:
                            rot, _ = R.align_vectors([direction], [z_axis])
                            quat = rot.as_quat()
                            data.mocap_quat[idx, :] = [quat[3], quat[0], quat[1], quat[2]]
                        else:
                            data.mocap_quat[idx, :] = [1, 0, 0, 0]
                        model.geom_size[moment_arrow_geom_ids[i]][1] = length / 2  # half-length


                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)
                    data.xfrc_applied[tool_body_id, :] = 0.0

                print("Final joint angles (deg):", np.round(np.degrees(data.qpos[:6]), 2))

            print("\n--- Finished all configurations. ---")
            input("Press Enter to close the viewer and exit...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

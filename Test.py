#!/usr/bin/env python3
import mujoco
import mujoco.viewer # Launch the graphical viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(roll, pitch, yaw, degrees = False):
    '''
    Function to convert euler angles (roll, pitch, yaw) to a quaternion.
    MuJoCo works with the w, x, y, z format.
    '''
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees = degrees)
    q = r.as_quat()  # Returns [x, y, z, w] format
    return [q[3], q[0], q[1], q[2]]  # Convert to [w, x, y, z] format

def main():

    # Path to the MuJoCo XML file
    xml_path = "C:/Users/chris/OneDrive - Politecnico di Milano/Politecnico di Milano/PhD - dottorato/GitHub repositories Lenovo/Screwdriving_MuJoCo/universal_robots_ur5e/scene.xml"
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path) # Load model
        data = mujoco.MjData(model) # data object to store dynamic state      
        mujoco.mj_resetData(model, data) # Reset the model to the initial state

        # Desired robot poses (in joints)
        target_qpos_list = [
            np.radians([100, -94.96, 101.82, -95.72, -96.35, -40.97]),
            np.radians([15.99, -62.07, 121.58, -159.82, -99, -90]),
            np.radians([-28.83, -84.36, 102.8, 18.69, -98.83, -101.46]),
            np.radians([-29.77, -160.55, 110.57, -186.16, -98.81, -101.15]),
            np.radians([-124.35, -158.41, 147, -153.21, -155.31, -98.24])
        ]

        # Viewer and simulation loop
        with mujoco.viewer.launch_passive(model, data) as viewer: # 'passive' => do not interact with the keyboard
            for iteration in range(5):

                # Get a target joint position
                desired_qpos = target_qpos_list[iteration]

                # Move base position               
                base_body_name = "base"  # Replace with the exact name of your robot's base body
                base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)

                # Set new position and orientation directly
                model.body_pos[base_body_id] = [iteration * 0.3, iteration * 0.1, 0.5]
                model.body_quat[base_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)
                

                # Set initial joint positions to match new target
                data.qpos[:6] = desired_qpos.copy()
                data.qvel[:6] = 0
                mujoco.mj_forward(model, data) # Compute derived quantities like transforms and forces

                # New part
                data.qacc[:] = 0
                gravity = np.zeros(model.nv)
                mujoco.mj_rne(model, data, 0, gravity) # Use recursive Newton-Euler method for inverse dynamics
                gravity = gravity[:6]
                print(f"The torques to compensate gravity at iteration {iteration} are: {gravity}")

                # Simulate for 2 seconds
                for _ in range(int(2.0 / model.opt.timestep)):
                    if not viewer.is_running():
                        break
                    for i in range(6):
                        data.ctrl[i] = gravity[i]

                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

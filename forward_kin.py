#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import threading

''' 
Use this code to test the forward kinematics, specifying:
- A target joint configuration
- The site name for the end-effector pose
'''

def main():
    base_dir = os.path.dirname(__file__)
    xml_path = os.path.join(base_dir, "universal_robots_ur5e/scene.xml")

    target_qpos = np.radians([113.05, -48.60, 95.07, -135.13, -89.28, 205.72])

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Set robot to the desired configuration
    data.qpos[:6] = target_qpos
    data.qvel[:6] = 0.0
    mujoco.mj_forward(model, data)

    # Compute gravity compensation torques
    data.qacc[:] = 0.0
    gravity = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, gravity)
    gravity_comp = gravity[:6]

    # ---- Get ee_site pose ----
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site") 
    pos = data.site_xpos[site_id]
    mat = data.site_xmat[site_id].reshape(3, 3)
    quat = R.from_matrix(mat).as_quat()  # Returns (x, y, z, w)

    print("\n--- ee_site Cartesian Pose ---")
    print("Position (x, y, z):", pos)
    print("Orientation (3x3 rotation matrix):\n", mat)
    print("Orientation (xyzw quaternion):", quat)
    print("-----------------------------\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Robot is being held at the target configuration with gravity compensation.")
        print("Press Enter to close the viewer and exit...")

        # Run until Enter is pressed
        running = True

        def wait_for_enter():
            nonlocal running
            input()
            running = False

        thread = threading.Thread(target=wait_for_enter, daemon=True)
        thread.start()

        while viewer.is_running() and running:
            data.ctrl[:6] = gravity_comp
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()

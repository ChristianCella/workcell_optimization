import mujoco
import numpy as np
from pathlib import Path
import os
import sys
from pathlib import Path
import mink
from mink.contrib import TeleopMocap

# Load the model and data
base_dir = os.path.dirname(__file__)
xml_path = os.path.join(base_dir, "universal_robots_ur5e/scene.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initial joint configuration (optional: use home, zero, or random)
initial_qpos = np.zeros(model.nq)
data.qpos[:] = initial_qpos
mujoco.mj_forward(model, data)

# === Define your desired pose ===
desired_position = np.array([0.5, 0.0, 0.7])  # meters (example)
desired_quat = np.array([1, 0, 0, 0])         # w, x, y, z (identity rotation)

# Build the SE3 target (mink has a helper for this)
wxyz_xyz = np.concatenate([desired_quat, desired_position])
T_target = mink.SE3(wxyz_xyz=wxyz_xyz)

# Create the configuration object
configuration = mink.Configuration(model)
configuration.update(data.qpos)

# Create the IK task for the site you want to move (e.g., "attachment_site")
ik_task = mink.FrameTask(
    frame_name="ee_site",  # or "tool_site", "ee_site", etc.
    frame_type="site",
    position_cost=1.0,
    orientation_cost=1.0,
    lm_damping=1e-6,
)

ik_task.set_target(T_target)

# Solve IK (single call)
tasks = [ik_task]
solved_vel = mink.solve_ik(
    configuration,
    tasks,
    dt=1.0,            # large dt so it goes "all the way"
    solver="osqp",     # "daqp" or "osqp" (make sure installed)
    limits=None,
)
configuration.integrate_inplace(solved_vel, 1.0)

# Print result
print("IK result joint angles (radians):", configuration.q)
print("IK result joint angles (degrees):", np.degrees(configuration.q))

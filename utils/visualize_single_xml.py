import mujoco
import mujoco.viewer
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from mujoco_utils import scene_manager

# Path to your XML file
ur5e_utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ur5e_utils_mujoco'))

# Retrieve the appropriate scene XML
model_path = scene_manager("full", 3, ur5e_utils_dir, "bringup_ur5e.xml", "extension.xml")

# Load the model
model = mujoco.MjModel.from_xml_path(model_path)

# Create data structure
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_forward(model, data)
    viewer.sync()
    input("Press Enter to continue...")
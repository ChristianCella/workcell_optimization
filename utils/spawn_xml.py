import mujoco
import mujoco.viewer
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Path to your XML file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_dir)
model_path = os.path.join(base_dir, "ur5e_utils_mujoco/fairino/fairino_FR10.xml")

# Load the model
model = mujoco.MjModel.from_xml_path(model_path)

# Create data structure
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_forward(model, data)
    viewer.sync()
    input("Press Enter to continue...")

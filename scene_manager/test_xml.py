import mujoco
import mujoco.viewer
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Path to your XML file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_dir)

#model_path = os.path.join(base_dir, "GoFa_utils_mujoco/GoFa5/empty_environment.xml")
model_path = os.path.join(base_dir, "GoFa_utils_mujoco/GoFa5/GoFa5.xml")

# Load the model
model = mujoco.MjModel.from_xml_path(model_path)

# Create data structure
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.sync()

    
    input("Press Enter to continue...")

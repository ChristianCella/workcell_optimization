import mujoco
import mujoco.viewer
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Path to your XML file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_dir)
# model_path = os.path.join(base_dir, "ur5e_utils_mujoco/screwing_tool/small_tool.xml")
# model_path = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/ur5e.xml")
# model_path = os.path.join(base_dir, "ur5e_utils_mujoco/empty_environment.xml")
# model_path = os.path.join(base_dir, "ur5e_utils_mujoco/obstacles/cube.xml")
model_path = os.path.join(base_dir, "ur5e_utils_mujoco/screwing_tool/screwdriver.xml")
# model_path = os.path.join(base_dir, "ur5e_utils_mujoco/screwing_pieces/table_grip.xml")
# model_path = os.path.join(base_dir, "GoFa_utils_mujoco/GoFa5/GoFa5.xml")

# Load the model
model = mujoco.MjModel.from_xml_path(model_path)

# Create data structure
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    q = np.array([0.9951, 4.5500, -1.5874, 0.5281, -4.0680, 2.2158])
    data.qpos[:6] = q.tolist()
    mujoco.mj_forward(model, data)
    viewer.sync()
    input("Press Enter to continue...")

import mujoco
import mujoco.viewer
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Path to your XML file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_dir)
# model_path = os.path.join(base_dir, "ur5e_utils_mujoco/screwing_tool/small_tool.xml")
# model_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/kuka/iiwa14.xml")
# model_path = os.path.join(base_dir, "ur5e_utils_mujoco/empty_environment.xml")
# model_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/simple_obstacles/Linear_guide.xml")
#model_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/screwing_tool/driller.xml")
model_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/screwing_pieces/aluminium_plate.xml")

# Load the model
model = mujoco.MjModel.from_xml_path(model_path)

# Create data structure
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit the viewer.")
    while viewer.is_running():
        viewer.sync()

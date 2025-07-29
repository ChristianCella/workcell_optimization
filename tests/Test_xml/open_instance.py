import mujoco
import mujoco.viewer
import os, sys

# Path to your XML file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_dir)
# model_path = os.path.join(base_dir, "ur5e_utils_mujoco/small_tool.xml")
model_path = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e_robot/ur5e.xml")

# Load the model
model = mujoco.MjModel.from_xml_path(model_path)

# Create data structure
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit the viewer.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

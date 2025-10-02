import mujoco
import mujoco.viewer
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Path to your XML file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_dir)

# Path to utils
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
from mujoco_utils import get_collisions

#model_path = os.path.join(base_dir, "GoFa_utils_mujoco/GoFa5/empty_environment.xml")
model_path = os.path.join(base_dir, "GoFa_utils_mujoco/GoFa12/GoFa12.xml")

# Load the model
model = mujoco.MjModel.from_xml_path(model_path)

# Create data structure
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    #q = np.radians([42, 20, 14, 0, 57, 72])
    #q = np.radians([53, -55, 84, -62, 138, 40]) # Example of colliding geometry
    #q = np.radians([0, -8, 70, 0, 25, 0])
    #q = np.array([2.4377, -0.2343, -1.2235, -0.3757,  0.1523, -1.3767])
    q = np.array([2.9064, -1.6322, -2.8301, -0.2380,  3.0534,  1.7904])
    data.qpos[:6] = q.tolist()
    mujoco.mj_forward(model, data)
    viewer.sync()

    # Get site id and world position (sites use xipos in MuJoCo v2)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    ee_pos = data.site_xpos[site_id]
    print(f"End-effector position: {ee_pos}")

    # Get collisions
    collisions = get_collisions(model, data, True)
    
    input("Press Enter to continue...")

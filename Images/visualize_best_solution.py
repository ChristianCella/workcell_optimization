import os, sys
import pandas as pd
import numpy as np
import mujoco
import mujoco.viewer

# Append the path to 'utils'
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
import fonts
from transformations import euler_to_quaternion, get_world_wrench, get_homogeneous_matrix
from mujoco_utils import set_body_pose


# Append the path to 'scene_manager'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from create_scene import create_scene

# Path setup
tool_filename = "screwdriver.xml"
robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
output_scene_filename = "final_scene.xml"
piece_name = "table_grip.xml" 
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Create the scene
model_path = create_scene(tool_name=tool_filename, robot_and_tool_file_name=robot_and_tool_file_name,
                            output_scene_filename=output_scene_filename, piece_name=piece_name, base_dir=base_dir)

# Load the newly created model
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# Load the file containing the optimal xi
csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_csv_path = os.path.join(csv_dir, "results/data/screwing/cma_es_ikflow/best_solutions.csv")
df = pd.read_csv(dataset_csv_path) 

# Print the last row (optimal result)
last = df.iloc[-1]
xi = last.values

print(f"{fonts.green}The complete vector is {xi}{fonts.reset}")

# Define the wrenches
local_wrenches = [
    (np.array([0, 0, -30, 0, 0, -20])),
    (np.array([0, 0, -30, 0, 0, -20])),
    (np.array([0, 0, -30, 0, 0, -20])),
    (np.array([0, 0, -30, 0, 0, -20])),
]

# Get body/site IDs
base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_grip")
ref_body_ids = []
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and name.startswith("hole_") and name.endswith("_frame_body"):
        ref_body_ids.append(i)
tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    input("Press enter to visualize the result ...")
    mujoco.mj_resetData(model, data)

    # Set robot base
    set_body_pose(model, data, base_body_id, [xi[0], xi[1], 0.1], euler_to_quaternion(xi[2], 0, 0)) 

    # Set piece
    set_body_pose(model, data, piece_body_id, [xi[6], xi[7], 0.0], euler_to_quaternion(0, 0, 0))

    # Set tool
    set_body_pose(model, data, screwdriver_body_id, [xi[3], xi[4], 0.03], euler_to_quaternion(xi[5], 0, 0))

    # Set robot joints
    q0_final = np.array([xi[8], xi[9], xi[10], xi[11], xi[12], xi[13]])
    data.qpos[:6] = q0_final.tolist()
    mujoco.mj_forward(model, data)
    viewer.sync()

    input(f"Press enter to continue")

    for idx in range(len(local_wrenches)):
        pass

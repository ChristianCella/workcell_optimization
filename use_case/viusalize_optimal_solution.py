import os, sys
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

#* Base directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

#* Scene manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from create_scene import create_scene
from parameters import UseCaseData, Ur5eRobot, Tools

#* Utils 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from transformations import rotm_to_quaternion, get_homogeneous_matrix, euler_to_quaternion
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, compute_jacobian
from ikflow_inference import FastIKFlowSolver, solve_ik_fast
import fonts

#* Initialize parameters
opt_par = UseCaseData()
rob_par = Ur5eRobot()
tool_par = Tools()

#* Constant matrices
_, _, A_wl3_ee = get_homogeneous_matrix(0, 0.1, 0, -90, 0, 0)
_, _, A_ee_t1 = get_homogeneous_matrix(-0.06, 0.0, tool_par.fixed_radius, 0.0, -90.0, 90.0)
_, _, A_t2_t = get_homogeneous_matrix(0, -0.195, 0.028, 90.0, 0.0, 0.0)

# Path setup 
tool_filename = "screwdriver_marco.xml"
robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
output_scene_filename = "final_scene.xml"
piece_name = "plate.xml" 

# Create the scene
model_path = create_scene(tool_name=tool_filename, robot_and_tool_file_name=robot_and_tool_file_name,
                            output_scene_filename=output_scene_filename, piece_name=piece_name, base_dir=base_dir)

# Load the newly created model
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# Load files containing optimal results
csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
layout_path = os.path.join(csv_dir, f"results", f"{opt_par.mode}", "all_joints", f"best_layout.csv")
configurations_path  = os.path.join(csv_dir, f"results", f"{opt_par.mode}", "all_joints",f"best_joints_configs.csv")

df_layout = pd.read_csv(layout_path)
df_configurations  = pd.read_csv(configurations_path)

last_layout = df_layout.iloc[-1]
last_configuration  = df_configurations.iloc[-1]

layout = last_layout.values.astype(float)
configuration = last_configuration.to_numpy(dtype=float) 

def parse_vec(s: str) -> np.ndarray:
    """Parse a string like '[a b c ...]' into a float vector."""
    return np.fromstring(str(s).replace('[', '').replace(']', ''), sep=' ', dtype=float)

# verify joint configuration shape
q_mat = configuration.reshape(-1, rob_par.nu)
assert q_mat.shape[1] == rob_par.nu, "best_joints_configs.csv joint count does not match robot_parameters.nu"

print(f"{fonts.green}The complete vector xi is {layout}{fonts.reset}")
print(f"{fonts.red}The complete vector q is:\n{q_mat}{fonts.reset}")

# Get body & site IDs
base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")

# Get the ids of the screws in the xml file
target_body_ids = []
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and name.startswith("hole_") and name.endswith("_frame_body"):
        target_body_ids.append(i)

#! Visualize the optimized workcell
with mujoco.viewer.launch_passive(model, data) as viewer:
    input("Press enter to visualize the result ...")
    mujoco.mj_resetData(model, data)

    # Robot base 
    set_body_pose(model, data, base_body_id, [0.0, 0.0, 0.0], euler_to_quaternion(0.0, 0.0, 0.0))

    # Workpiece
    _, _, A_w_p = get_homogeneous_matrix(layout[0], layout[1], -0.02, 0.0, 0.0, 0.0) 
    set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

    # Rotated frame
    theta = layout[2]
    _, _, A_t1_t2 = get_homogeneous_matrix(0.0, tool_par.fixed_radius - (tool_par.fixed_radius * np.cos(theta)), -tool_par.fixed_radius * np.sin(theta), np.degrees(theta), 0.0, 0.0)
    A_ee_t2 = A_ee_t1 @ A_t1_t2
    set_body_pose(model, data, screwdriver_body_id, A_ee_t2[:3, 3], rotm_to_quaternion(A_ee_t2[:3, :3]))

    # Set the tip frame
    A_ee_t = A_ee_t1 @ A_t1_t2 @ A_t2_t
    set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

    # Wrench
    world_wrench = [(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -2.0]))]

    # Set robot home configuration
    q0 = rob_par.home_configuration.copy()
    data.qpos[:rob_par.nu] = q0.tolist()
    mujoco.mj_forward(model, data)
    viewer.sync()

    norms = []
    input("Press enter to continue")

    for idx in range(len(target_body_ids)):

        # Apply best joints for this piece
        data.qpos[:rob_par.nu] = q_mat[idx].tolist()
        data.qvel[:] = 0
        data.qacc[:] = 0
        data.ctrl[:] = 0
        mujoco.mj_forward(model, data)
        viewer.sync()
        
        # Jacobian at tool_site and frame conversions
        J = compute_jacobian(model, data, tool_site_id)
        tau_g = data.qfrc_bias[:rob_par.nu]
        tau_ext = J.T @ world_wrench[0]
        tau_tot = (tau_ext + tau_g) / (rob_par.gear_ratios * rob_par.max_torques)
        norms.append(np.max(tau_tot[-3:]))

        print(f"{fonts.blue}External torques: {np.array2string(tau_ext, precision=2)}{fonts.reset}")
        print(f"{fonts.red}Gravity torques:  {np.array2string(tau_g,   precision=2)}{fonts.reset}")       
        input(f"Press Enter to see the next piece configuration (piece {idx+1})…")

    print(f"f obtained testing the layout: {float(np.mean(norms))}")
import os, sys
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer

# Append the path to 'utils'
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
import fonts
from transformations import (
    euler_to_quaternion,
    rotm_to_quaternion,
    get_world_wrench,
    get_homogeneous_matrix,
)
from mujoco_utils import set_body_pose, compute_jacobian

# Append the path to 'scene_manager'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from create_scene import create_scene
from parameters import ScrewingCMAES, Ur5eRobot

parameters = ScrewingCMAES()
robot_parameters = Ur5eRobot()

# -----------------------------
# Paths and scene creation
# -----------------------------
tool_filename = "screwdriver.xml"
robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
output_scene_filename = "final_scene.xml"
piece_name = "table_grip.xml"
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Create the scene XML and load it
model_path = create_scene(
    tool_name=tool_filename,
    robot_and_tool_file_name=robot_and_tool_file_name,
    output_scene_filename=output_scene_filename,
    piece_name=piece_name,
    base_dir=base_dir,
)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# -----------------------------
# Load CSVs (last/best row)
# -----------------------------
csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
xi_path = os.path.join(csv_dir, "results/data/screwing/turbo_ikflow/best_solutions.csv")
q_path  = os.path.join(csv_dir, "results/data/screwing/turbo_ikflow/best_configs.csv")

df_xi = pd.read_csv(xi_path)
df_q  = pd.read_csv(q_path)

last_xi = df_xi.iloc[-1]
last_q  = df_q.iloc[-1]

# xi = [x_b, y_b, theta_x_b, x_t, y_t, theta_x_t, x_p, y_p, q01..q06]
xi = last_xi.values.astype(float)

def parse_vec(s: str) -> np.ndarray:
    """Parse a string like '[a b c ...]' into a float vector."""
    return np.fromstring(str(s).replace('[', '').replace(']', ''), sep=' ', dtype=float)

# q_mat: shape (n_pieces, nu) e.g. (4, 6), columns order in df_q is config_0, config_1, ...
q_mat = np.vstack([parse_vec(last_q[c]) for c in df_q.columns])
assert q_mat.shape[1] == robot_parameters.nu, "best_configs.csv joint count does not match robot_parameters.nu"

print(f"{fonts.green}The complete vector xi is {xi}{fonts.reset}")
print(f"{fonts.red}The complete vector q is:\n{q_mat}{fonts.reset}")

# -----------------------------
# Wrenches (tool frame)
# -----------------------------
local_wrenches = [
    np.array([0, 0, -30, 0, 0, -20]),
    np.array([0, 0, -30, 0, 0, -20]),
    np.array([0, 0, -30, 0, 0, -20]),
    np.array([0, 0, -30, 0, 0, -20]),
]

# -----------------------------
# IDs
# -----------------------------
base_body_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
tool_body_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")   # tool tip body (owner of tool_site)
piece_body_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_grip")
tool_site_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")     # tool top (t1)

# -----------------------------
# Viewer & replay
# -----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    input("Press enter to visualize the result ...")
    mujoco.mj_resetData(model, data)

    # Base, piece, tool-top (t1)
    set_body_pose(model, data, base_body_id,  [xi[0], xi[1], 0.1], euler_to_quaternion(xi[2], 0, 0))
    set_body_pose(model, data, piece_body_id, [xi[6], xi[7], 0.0], euler_to_quaternion(0, 0, 0))
    set_body_pose(model, data, screwdriver_body_id, [xi[3], xi[4], 0.03], euler_to_quaternion(xi[5], 0, 0))

    # IMPORTANT: reconstruct tool-tip (t) pose from tool-top (t1) using the SAME fixed offset used in optimization
    # Build A_ee_t1 from xi[3:6] (get_homogeneous_matrix expects degrees for rotations)
    _, _, A_ee_t1 = get_homogeneous_matrix(float(xi[3]), float(xi[4]), 0.03, np.degrees(float(xi[5])), 0, 0)
    # Fixed transform from tool-top (t1) to tool-tip (t)
    _, _, A_t1_t  = get_homogeneous_matrix(0, 0, 0.32, 0, 0, 0)
    # Final tool-tip pose
    A_ee_t = A_ee_t1 @ A_t1_t
    set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

    # Set robot home joints from xi (not strictly required for the per-piece replay)
    q0_final = np.array([xi[8], xi[9], xi[10], xi[11], xi[12], xi[13]], dtype=float)
    data.qpos[:robot_parameters.nu] = q0_final.tolist()
    mujoco.mj_forward(model, data)
    viewer.sync()

    norms = []
    input("Press enter to continue")

    for idx in range(len(local_wrenches)):
        print(f"Layout for piece {idx+1}")

        # Apply best joints for this piece
        data.qpos[:robot_parameters.nu] = q_mat[idx].tolist()
        data.qvel[:] = 0
        data.qacc[:] = 0
        data.ctrl[:] = 0
        mujoco.mj_forward(model, data)
        viewer.sync()

        # Jacobian at tool_site and frame conversions
        J = compute_jacobian(model, data, tool_site_id)                     # (6 x nu)
        tau_g = data.qfrc_bias[:robot_parameters.nu]                        # gravity torques
        R_tool_to_world = data.site_xmat[tool_site_id].reshape(3, 3)
        R_world_to_tool = R_tool_to_world.T
        world_wrench = get_world_wrench(R_world_to_tool, local_wrenches[idx])
        tau_ext = J.T @ world_wrench
        tau_tot = tau_g + tau_ext

        print(f"{fonts.blue}External torques: {np.array2string(tau_ext, precision=6)}{fonts.reset}")
        print(f"{fonts.red}Gravity torques:  {np.array2string(tau_g,   precision=6)}{fonts.reset}")

        # Normalize by gear ratios & motor limits (same normalization used in optimization)
        norms.append(
            np.linalg.norm(
                tau_tot / (np.array(robot_parameters.gear_ratios) * np.array(robot_parameters.max_torques))
            )
        )
        input(f"Press Enter to see the next piece configuration (piece {idx+1})…")

    print(f"f obtained testing the layout: {float(np.mean(norms))}")

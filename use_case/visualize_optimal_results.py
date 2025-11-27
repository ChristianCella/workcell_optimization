import os, sys
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer

# Append the path to 'utils'
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
import fonts
from constant_parameters import OptimizationParameters, Ur5eRobot
from transformations import euler_to_quaternion, rotm_to_quaternion, get_world_wrench, get_homogeneous_matrix
from mujoco_utils import set_body_pose, compute_jacobian, inverse_manipulability

parameters = OptimizationParameters()
robot_parameters = Ur5eRobot()

# Path setup 
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
model_path = os.path.join(base_dir, "ur5e_utils_mujoco/scene_ur5e.xml")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# Load files containing optimal results
csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
xi_path = os.path.join(csv_dir, "results/optimization/best_solutions.csv")
q_path  = os.path.join(csv_dir, "results/optimization/best_configs.csv")

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
    np.array([0, 0, 30, 0, 0, 20]),
    np.array([0, 0, 30, 0, 0, 20]),
    np.array([0, 0, 30, 0, 0, 20]),
    np.array([0, 0, 30, 0, 0, 20]),
]

# Get body/site IDs
base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
tool_base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base") # Base of the tool
tool_base_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_base_site')
tool_tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_tip")
tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_tip_site')
wrist_3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

#! Visualize the optimized workcell
with mujoco.viewer.launch_passive(model, data) as viewer:
    input("Press enter to visualize the result ...")
    mujoco.mj_resetData(model, data)

    # Base, piece, tool-top (t1)
    set_body_pose(model, data, base_body_id,  [xi[0], xi[1], 0.1], euler_to_quaternion(xi[2], 0, 0))
    set_body_pose(model, data, tool_base_body_id, [xi[3], xi[4], 0.03], euler_to_quaternion(xi[5], 0, 0))

    # Reconstruct the tool tip
    _, _, A_ee_t1 = get_homogeneous_matrix(float(xi[3]), float(xi[4]), 0.03, np.degrees(float(xi[5])), 0, 0)
    _, _, A_t1_t  = get_homogeneous_matrix(0, 0, 0.32, 0, 0, 0) #* Always the same offset
    A_ee_t = A_ee_t1 @ A_t1_t
    set_body_pose(model, data, tool_tip_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

    # Set robot home joints from xi (not strictly required for the per-piece replay)
    q0_final = np.array([xi[8], xi[9], xi[10], xi[11], xi[12], xi[13]], dtype=float)
    data.qpos[:robot_parameters.nu] = q0_final.tolist()
    mujoco.mj_forward(model, data)
    viewer.sync()

    #! S = H^-T * Gamma^-T * Gamma^-1 * H^-1
    gear_ratios = robot_parameters.gear_ratios
    max_torques = robot_parameters.max_torques
    H_mat = np.diag(gear_ratios)
    Gamma_mat = np.diag(max_torques)
    S = np.linalg.inv(H_mat.T) @ np.linalg.inv(Gamma_mat.T) @ np.linalg.inv(Gamma_mat) @ np.linalg.inv(H_mat) 

    # centers and half-ranges
    plan_joint_ids = np.arange(robot_parameters.nu, dtype=int)
    jnt_range = model.jnt_range[plan_joint_ids].copy()
    lb = jnt_range[:, 0]
    ub = jnt_range[:, 1]
    centers = (lb + ub) / 2
    half_ranges = (ub - lb) / 2
    norms = []
    input("Press enter to continue")

    for idx in range(len(local_wrenches)):

        # Apply best joints for this piece
        data.qpos[:robot_parameters.nu] = q_mat[idx].tolist()
        data.qvel[:] = 0
        data.qacc[:] = 0
        data.ctrl[:] = 0
        mujoco.mj_forward(model, data)
        viewer.sync()

        # Jacobian at tool_site and frame conversions
        J = compute_jacobian(model, data, tool_tip_site_id)
        tau_g = data.qfrc_bias[:robot_parameters.nu] #* Torques due to gravity
        R_tool_to_world = data.site_xmat[tool_tip_site_id].reshape(3, 3)
        R_world_to_tool = R_tool_to_world.T
        world_wrench = get_world_wrench(R_world_to_tool, local_wrenches[idx])
        tau_ext = J.T @ world_wrench
        tau_tot = tau_g + tau_ext
        tau_joint_lim = gear_ratios * max_torques

        # Compute alpha, beta and gammma
        alpha = world_wrench.T @ J @ S @ J.T @ world_wrench
        beta = 2 * world_wrench.T @ J @ S @ tau_g
        gamma = tau_g.T @ S @ tau_g

        # Compute lambdas with the formula
        lambda1 = (-beta + np.sqrt(beta**2 + 4 * alpha * (1 - gamma))) / (2 * alpha)
        lambda2 = (-beta - np.sqrt(beta**2 + 4 * alpha * (1 - gamma))) / (2 * alpha)
        lambda_star = np.max([lambda1, lambda2]) 

        # Compute the real ratios for scaling
        lambda_real = np.abs(tau_joint_lim) / np.abs(tau_tot)

        # Compute the manipulability
        f_delta_j = inverse_manipulability(q_mat[idx].tolist().copy(), model, data, tool_tip_site_id)

        # Verify 'how centered' the robot is
        z = (data.qpos[:robot_parameters.nu] - centers) / half_ranges
        s_val = 1.0 - np.abs(z)
        s_mean = np.mean(s_val)

        print(f"{fonts.blue}External torques: {np.array2string(tau_ext, precision=6)}{fonts.reset}")
        print(f"{fonts.red}Gravity torques:  {np.array2string(tau_g,   precision=6)}{fonts.reset}")
        print(f"{fonts.green}Lambda_max for target {idx+1} is {lambda_star}{fonts.reset}")
        print(f"{fonts.cyan}Minimum scaling ratio for target {idx+1} is {np.min(lambda_real)} (joint {np.argmin(lambda_real)}){fonts.reset}")
        print(f"{fonts.yellow}Manipulability for target {idx+1} is {f_delta_j}{fonts.reset}")
        print(f"{fonts.purple}Joint centering measure s for target {idx+1} is {s_mean}{fonts.reset}")

        # Normalize by gear ratios & motor limits (same normalization used in optimization)
        norms.append(
            np.linalg.norm(
                tau_tot / (np.array(robot_parameters.gear_ratios) * np.array(robot_parameters.max_torques))
            )
        )
        input(f"Press Enter to see the next piece configuration (piece {idx+1})…")

    print(f"f obtained testing the layout: {float(np.mean(norms))}")
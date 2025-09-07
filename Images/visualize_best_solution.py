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
from mujoco_utils import set_body_pose, compute_jacobian, inverse_manipulability

# Append the path to 'scene_manager'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from create_scene import create_scene
from parameters import ScrewingCMAES, KukaIiwa14

parameters = ScrewingCMAES()
robot_parameters = KukaIiwa14()

def parse_vec(s: str) -> np.ndarray:
    """Parse a string like '[1 2 3]' into a float vector."""
    return np.fromstring(str(s).replace('[', '').replace(']', ''), sep=' ', dtype=float)


# -----------------------------
# Paths and scene creation
# -----------------------------
tool_filename = "driller.xml"
robot_and_tool_file_name = "temp_kuka_with_tool.xml"
output_scene_filename = "final_scene.xml"
piece_name1 = "aluminium_plate.xml" 
piece_name2 = "Linear_guide.xml"
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

model_path = create_scene(
    tool_name=tool_filename, 
    robot_and_tool_file_name=robot_and_tool_file_name,
    output_scene_filename=output_scene_filename, 
    piece_name1=piece_name1, 
    piece_name2=piece_name2, 
    base_dir=base_dir
)

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# -----------------------------
# Load CSVs
# -----------------------------
csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

xi_files = [
    os.path.join(csv_dir, "results/data/hole1/turbo_ikflow/best_solutions.csv"),
    os.path.join(csv_dir, "results/data/hole2/turbo_ikflow/best_solutions.csv"),
]
q_files = [
    os.path.join(csv_dir, "results/data/hole1/turbo_ikflow/best_configs.csv"),
    os.path.join(csv_dir, "results/data/hole2/turbo_ikflow/best_configs.csv"),
]

# Load and stack xi
xi_list = []
for path in xi_files:
    df = pd.read_csv(path)
    xi_list.append(df.iloc[-1].to_numpy())  # Convert Series to np.array

xi = np.vstack(xi_list)  # Shape: (n_pieces, 3)

# Load and stack q
q_list = []
for path in q_files:
    df = pd.read_csv(path)
    q_strs = df.iloc[-1].values
    q_parsed = np.hstack([parse_vec(s) for s in q_strs])
    q_list.append(q_parsed)

q_mat = np.vstack(q_list)  # Shape: (n_pieces, nu)

print(f"{fonts.green}The complete vector xi is:\n{xi}{fonts.reset}")
print(f"{fonts.red}The complete vector q is:\n{q_mat}{fonts.reset}")

# -----------------------------
# Wrenches
# -----------------------------
local_wrenches = [
    np.array([0, 0, -150, 0, 0, -10]),
    np.array([0, 0, -150, 0, 0, -10]),
]

# -----------------------------
# IDs
# -----------------------------
base_body_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
tool_body_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
piece_body_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "aluminium_plate")
tool_site_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")

# NOTE: Temporarily overwrite q_mat

q_mat = np.array([[-1.28, 1.383, 1.558, -0.396, 1.668, -1.725, -2.45], 
                  [-2.047, 1.381, -1.655, -1.035, 1.449, 1.739, -1.529]])


# -----------------------------
# Viewer & replay
# -----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    viewer.sync()
    input("Press enter to visualize the result ...")
    mujoco.mj_resetData(model, data)

    for i in range(len(xi)):
        y_b, z_t, theta_y_p = xi[i]

        # Set base and piece poses
        #set_body_pose(model, data, base_body_id, [0.0, y_b, 0.15], euler_to_quaternion(0, 0, np.pi/2)) 
        set_body_pose(model, data, base_body_id, [0.0, 0.0, 0.15], euler_to_quaternion(0, 0, np.pi/2)) 
        #set_body_pose(model, data, piece_body_id, [0.6, 0.0, 0.8], euler_to_quaternion(0, theta_y_p, np.pi))
        set_body_pose(model, data, piece_body_id, [0.6, 0.0, 0.2], euler_to_quaternion(0, 0, np.pi))
        #set_body_pose(model, data, screwdriver_body_id, [0.1, 0.0, z_t], euler_to_quaternion(0, 0, 0))
        set_body_pose(model, data, screwdriver_body_id, [0.1, 0.0, 0.0], euler_to_quaternion(0, 0, 0))

        # Compute transformation tool_top → tool_frame
        _, _, A_ee_t1 = get_homogeneous_matrix(0.1, 0.0, 0, 0, 0, 0)
        _, _, A_t1_t  = get_homogeneous_matrix(0, 0, 0.26, 0, 0, 0)
        A_ee_t = A_ee_t1 @ A_t1_t
        set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

        # Set robot to home joint config (optional)
        q0_final = np.array([2.014, 0.201, -0.098, 1.735, -0.018, -1.607, 0.346])
        data.qpos[:robot_parameters.nu] = q0_final
        mujoco.mj_forward(model, data)
        viewer.sync()

        input("Press Enter to continue")

        # Apply best joints for this layout
        data.qpos[:robot_parameters.nu] = q_mat[i]
        data.qvel[:] = 0
        data.qacc[:] = 0
        data.ctrl[:] = 0
        mujoco.mj_forward(model, data)
        viewer.sync()

        # Compute matrix S
        gear_ratios = robot_parameters.gear_ratios
        max_torques = robot_parameters.max_torques
        H_mat = np.diag(gear_ratios)
        Gamma_mat = np.diag(max_torques)
        S = np.linalg.inv(H_mat.T) @ np.linalg.inv(Gamma_mat.T) @ np.linalg.inv(Gamma_mat) @ np.linalg.inv(H_mat) #! S = H^-T * Gamma^-T * Gamma^-1 * H^-1

        # Compute torques
        J = compute_jacobian(model, data, tool_site_id)
        tau_g = data.qfrc_bias[:robot_parameters.nu]
        R_tool_to_world = data.site_xmat[tool_site_id].reshape(3, 3)
        R_world_to_tool = R_tool_to_world.T
        world_wrench = get_world_wrench(R_world_to_tool, local_wrenches[i])
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
        f_delta_j = inverse_manipulability(q_mat[i].tolist().copy(), model, data, tool_site_id)

        print(f"{fonts.blue}External torques: {np.array2string(tau_ext, precision=6)}{fonts.reset}")
        print(f"{fonts.red}Gravity torques:  {np.array2string(tau_g,   precision=6)}{fonts.reset}")
        print(f"{fonts.green}Lambda_max is {lambda_star}{fonts.reset}")
        print(f"{fonts.cyan}Minimum scaling ratiois {np.min(lambda_real)} (joint {np.argmin(lambda_real)}){fonts.reset}")
        print(f"{fonts.yellow}Manipulability is {f_delta_j}{fonts.reset}")


        norm = np.linalg.norm(
            tau_tot / (np.array(robot_parameters.gear_ratios) * np.array(robot_parameters.max_torques))
        )
        print(f"{fonts.green}Normalized torque norm for piece {i+1}: {norm:.4f}{fonts.reset}")

        input(f"Press Enter to proceed to the next piece...\n")

    print("Simulation finished.")

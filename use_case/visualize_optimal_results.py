import os, sys
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

#* ur5e 
ur5e_utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ur5e_utils_mujoco'))

#* Utils
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
import fonts
from constant_parameters import OptimizationParameters, Ur5eRobot, Tools
from transformations import euler_to_quaternion, rotm_to_quaternion, get_homogeneous_matrix, quaternion_to_euler
from mujoco_utils import set_body_pose, compute_jacobian, scene_manager, get_cartesian_pose

#* Database 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../database')))
from query_db import complete_query

#* Initialize parameters
opt_par = OptimizationParameters()
rob_par = Ur5eRobot()
tool_par = Tools()

#* Constant matrices
_, _, A_wl3_ee = get_homogeneous_matrix(0, 0.1, 0, -90, 0, 0)
_, _, A_eb_et = get_homogeneous_matrix(0, 0, tool_par.extension_offset, 0, 0, 0)

# Query the database
n_targets, targets_poses, world_wrenches, tool_ids, clusters = complete_query()
world_wrenches = [np.array(w) if w is not None else np.zeros(6) for w in world_wrenches]

if opt_par.theta == 1: #! One single optimization
    list_n_targets       = [n_targets]
    list_target_poses    = [targets_poses]
    list_world_wrenches  = [world_wrenches]
    list_tool_ids       = [tool_ids]
    print(f"The optimziation lasts for {len(list_n_targets)} iterations")

elif opt_par.theta == 0: #! Optimization for clusters of targets
    #* Organize targets by clusters
    unique_clusters = sorted(set(clusters)) 
    cluster_idx = {c: i for i, c in enumerate(unique_clusters)}

    list_n_targets       = [0] * len(unique_clusters)  
    list_target_poses    = [[] for _ in unique_clusters]
    list_world_wrenches  = [[] for _ in unique_clusters]
    list_tool_ids       = [[] for _ in unique_clusters]

    for pose, wrench, cl, id in zip(targets_poses, world_wrenches, clusters, tool_ids):
        idx = cluster_idx[cl]         
        list_n_targets[idx] += 1
        list_target_poses[idx].append(pose)
        list_world_wrenches[idx].append(wrench)
        list_tool_ids[idx].append(id)
    print(f"The optimziation lasts for {len(list_n_targets)} iterations")
else:
    raise ValueError("opt_par.theta must be either 0 or 1.")

# Cluster you want to visualize
cluster_to_visualize = 1

# Path setup 
model_path = scene_manager("full", list_n_targets[cluster_to_visualize - 1], ur5e_utils_dir, "bringup_ur5e.xml", "extension.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# Load files containing optimal results
csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
layout_path = os.path.join(csv_dir, f"results", f"{opt_par.mode}",f"best_layout_cluster_{cluster_to_visualize}.csv")
configurations_path  = os.path.join(csv_dir, f"results", f"{opt_par.mode}",f"best_joints_configs_cluster_{cluster_to_visualize}.csv")

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

# Get body/site IDs
base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
tool_base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base") 
tool_tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_tip")
tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_tip_site')
ext_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ext_base") 

# Get the ids of all the target locations
target_body_ids = []
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and name.startswith("reference_target_"):
        target_body_ids.append(i)

target_poses = list_target_poses[cluster_to_visualize - 1]

# Place the static targets in the scene
for i, body_id in enumerate(target_body_ids): 
    tetax, tetay, tetaz = quaternion_to_euler([target_poses[i][6], target_poses[i][3], target_poses[i][4], target_poses[i][5]], degrees=False)
    A_w_p = np.eye(4)
    A_w_p[:3, 3] = np.array([target_poses[i][0], target_poses[i][1], target_poses[i][2]])
    A_w_p[:3, :3] = R.from_euler('XYZ', [tetax, tetay, tetaz], degrees=False).as_matrix()
    set_body_pose(model, data, body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

#! Visualize the optimized workcell
with mujoco.viewer.launch_passive(model, data) as viewer:
    input("Press enter to visualize the result ...")
    mujoco.mj_resetData(model, data)

    # Robot base and tool base
    set_body_pose(model, data, base_body_id,  [layout[0], layout[1], 0], euler_to_quaternion(0, 0, 0))
    _, _, A_ee_t1 = get_homogeneous_matrix(0, 0, 0, 0, 0, 0)
    set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

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

        tool_id = list_tool_ids[cluster_to_visualize - 1][idx]
        if tool_id == "gripper_hande":
            gripper_length = tool_par.hande_offset

            # In case of gripper alone, move the extension away
            _, _, A_w_et = get_homogeneous_matrix(tool_par.detachment_pose[0], tool_par.detachment_pose[1], tool_par.detachment_pose[2], tool_par.detachment_pose[3], tool_par.detachment_pose[4], tool_par.detachment_pose[5])
            A_w_eb = A_w_et @ np.linalg.inv(A_eb_et)
            set_body_pose(model, data, ext_base_body_id, A_w_eb[:3, 3], rotm_to_quaternion(A_w_eb[:3, :3]))
            mujoco.mj_forward(model, data)
        elif tool_id == "FingerTool":   
            gripper_length = tool_par.extension_offset + tool_par.hande_offset

        # Set tip frame
        _, _, A_t1_t = get_homogeneous_matrix(0, 0, gripper_length, 0, 0, 0)
        set_body_pose(model, data, tool_tip_body_id, A_t1_t[:3, 3], rotm_to_quaternion(A_t1_t[:3, :3])) 
        A_ee_t = A_ee_t1 @ A_t1_t
        mujoco.mj_forward(model, data)

        # Update the pose of the extension, if needed
        if tool_id == "FingerTool":
            pos, eul = get_cartesian_pose(tool_tip_body_id, data, 'euler')
            _, _, A_w_et = get_homogeneous_matrix(pos[0], pos[1], pos[2], np.degrees(eul[0]), np.degrees(eul[1]), np.degrees(eul[2]))
            A_w_eb = A_w_et @ np.linalg.inv(A_eb_et)
            set_body_pose(model, data, ext_base_body_id, A_w_eb[:3, 3], rotm_to_quaternion(A_w_eb[:3, :3]))
            mujoco.mj_forward(model, data)
        viewer.sync()
        
        # Jacobian at tool_site and frame conversions
        J = compute_jacobian(model, data, tool_tip_site_id)
        tau_g = data.qfrc_bias[:rob_par.nu]
        tau_ext = J.T @ list_world_wrenches[cluster_to_visualize - 1][idx][:]
        tau_tot = (tau_ext + tau_g) / (rob_par.gear_ratios * rob_par.max_torques)
        norms.append(np.linalg.norm(tau_tot))

        print(f"{fonts.blue}External torques: {np.array2string(tau_ext, precision=6)}{fonts.reset}")
        print(f"{fonts.red}Gravity torques:  {np.array2string(tau_g,   precision=6)}{fonts.reset}")       
        input(f"Press Enter to see the next piece configuration (piece {idx+1})…")

    print(f"f obtained testing the layout: {float(np.mean(norms))}")
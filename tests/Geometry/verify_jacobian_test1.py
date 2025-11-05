import mujoco
import mujoco.viewer
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

''' 
First test: suppose to start from the Jacobian from base to ee, expressed in base frame.
Well, in mujoco, this can be computed as Ad(inv(A^w_b)) * J^{w}_{b->ee}, since the API gives J^{w}_{b->ee}.
Then, to compute the Jacobian from base to tool (a point offset from ee), expressed in world frame, we do:
1) Change expression frame from base to world: J^{w}_{b->ee} = Ad(A^w_b) * J^{b}_{b->ee}
2) Point shift from ee to tool: J^{w}_{b->t} = X^{w}_{t<-ee} * J^{w}_{b->ee} (with {t<-ee} meaning "shift from ee to tool")
Finally, wwe verify we obtain the same result as the API call J^{w}_{b->t}.
'''

# ---------- helpers for v–ω ordering ([v; ω]) ----------

def rot_from_xmat(xmat_flat):
    return xmat_flat.reshape(3, 3)

def skew(p):
    return np.array([[0, -p[2],  p[1]],
                     [p[2],  0, -p[0]],
                     [-p[1], p[0],  0]])

def tilde_Ad(T):
    """
    Adjoint for twists in v–ω ordering.
    Change expression frame of Jacobian: J^X = Ad^X_Y J^Y
    T = [[R, p],[0,1]] is pose of frame j in i.
    """
    R_ = T[:3, :3]
    p  = T[:3,  3]
    A = np.zeros((6,6))
    A[:3,:3] = R_
    A[:3,3:] = -skew(p) @ R_
    A[3:,3:] = R_
    return A

def tilde_X_point_shift(r):
    """
    Point-shift matrix in v–ω ordering.
    Shift twist from origin o to o' = o + r in SAME frame.
    """
    X = np.eye(6)
    X[:3,3:] = -skew(r)
    return X

def jac_site_world_vw(model, data, site_id):
    """
    MuJoCo gives Jp (linear), Jr (angular), both expressed in world.
    Stack as [Jp; Jr] -> v–ω ordering.
    """
    nv = model.nv
    Jp = np.zeros((3, nv))
    Jr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, site_id)
    return np.vstack([Jp, Jr])   # v–ω

def T_from_pos_rot(p, R_):
    T = np.eye(4)
    T[:3,:3] = R_
    T[:3, 3] = p
    return T

# -------------------------------------------------------

# Paths
base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_dir)

utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils'))
sys.path.append(utils_dir)

from mujoco_utils import get_collisions, set_body_pose
from transformations import rotm_to_quaternion

model_path = os.path.join(base_dir, "ur5e_utils_mujoco/final_scene.xml")

# Load model & data
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)

# IDs
base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
tool_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
tool_site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
sdriver_body  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")
ee_site_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # --- 1) Set base pose ---
    t_w_b = np.array([0.2, 0.2, 0.4])
    R_w_b = R.from_euler('XYZ', [0,0,0]).as_matrix()
    A_w_b = T_from_pos_rot(t_w_b, R_w_b)
    set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

    # --- 2) Place screwdriver top relative to ee ---
    t_ee_t1 = np.array([0, 0, 0.15])
    R_ee_t1 = np.eye(3)
    A_ee_t1 = T_from_pos_rot(t_ee_t1, R_ee_t1)
    set_body_pose(model, data, sdriver_body, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

    # fixed transform t1->t
    t_t1_t  = np.array([0, 0, 0.31])
    R_t1_t  = np.eye(3)
    A_t1_t  = T_from_pos_rot(t_t1_t, R_t1_t)

    # visualization: tool body
    A_ee_t = A_ee_t1 @ A_t1_t
    set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

    # --- 3) Set joints ---
    q = np.array([0.9951, 4.5500, -1.5874, 0.5281, -4.0680, 2.2158])
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    viewer.sync()

    # --- 4) Retrieve actual poses from MuJoCo ---
    R_w_b_true = rot_from_xmat(data.xmat[base_body_id])
    p_w_b_true = data.xpos[base_body_id]
    A_w_b_true = T_from_pos_rot(p_w_b_true, R_w_b_true)
    A_b_w_true = np.linalg.inv(A_w_b_true)

    p_w_ee = data.site_xpos[ee_site_id]
    p_w_t  = data.site_xpos[tool_site_id]
    r_w_t_ee = p_w_t - p_w_ee   # displacement ee->tool in world

    # --- 5) Jacobians from API ---
    J_w_t_api = jac_site_world_vw(model, data, tool_site_id)   # J^{w}_{b->t} v–ω
    J_w_ee_api = jac_site_world_vw(model, data, ee_site_id)    # J^{w}_{b->ee} v–ω
    print("J^{w}_{b->t} (API, v–ω):\n", J_w_t_api)

    # --- 6) Convert J^{w}_{b->ee} to J^{b}_{b->ee} ---
    J_b_ee = tilde_Ad(A_b_w_true) @ J_w_ee_api
    print("J^{b}_{b->ee} (v–ω):\n", J_b_ee)

    # --- 7) Recompute J^{w}_{b->t} geometrically ---
    # (a) base->world change of expression frame
    J_w_ee_geo = tilde_Ad(A_w_b_true) @ J_b_ee
    # (b) shift ee->t
    X_w_t_from_ee = tilde_X_point_shift(r_w_t_ee)
    J_w_t_geo = X_w_t_from_ee @ J_w_ee_geo
    print("J^{w}_{b->t} (geometric, v–ω):\n", J_w_t_geo)

    print("Max abs diff (geo - api):", np.max(np.abs(J_w_t_geo - J_w_t_api)))

    # --- 8) Example torque from wrench at tool in world (v–ω) ---
    F_w_tool = np.array([0.0, 0.0, -20.0,   0.0, 0.0, 0.5])  # [force; moment]
    tau = J_w_t_api.T @ F_w_tool
    print("Joint torques from wrench:", tau)

    collisions = get_collisions(model, data, True)
    input("Press Enter to continue...")
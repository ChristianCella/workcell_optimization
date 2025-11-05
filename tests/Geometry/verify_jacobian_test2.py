import mujoco
import mujoco.viewer
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

'''
Second test: we need to verify that the APIs give us the same result we should obtain by geometric reasoning.
We express all the z axis and the positions in world frame, then build the Jacobian columns as per textbook formulas.
For revolute joint i:   J_v = z_i x (p_ref - p_i),  J_ω = z_i
Where p_ref is the point where we want the Jacobian (ee or tool), and p_i, z_i are the joint position and axis in world frame.
In the end, this should be the same as the one from world to end effector, premultiplied by the point-shift matrix if needed.
'''

# ---------- helpers for v–ω ordering ([v; ω]) ----------

def rot_from_xmat(xmat_flat):
    return xmat_flat.reshape(3, 3)

def skew(p):
    return np.array([[0, -p[2],  p[1]],
                     [p[2],  0, -p[0]],
                     [-p[1], p[0],  0]])

def tilde_Ad(T):
    """Adjoint for twists in v–ω ordering (change expression frame)."""
    R_ = T[:3, :3]
    p  = T[:3,  3]
    A = np.zeros((6,6))
    A[:3,:3] = R_
    A[:3,3:] = -skew(p) @ R_
    A[3:,3:] = R_
    return A

def X_point_shift_vw(r):
    """
    Point-shift in v–ω ordering (same expression frame).
    Using r = p_ref(new) - p_ref(old).
    For twists:  [v; ω]_new = [I, [r]x; 0, I] [v; ω]_old
    Hence for Jacobians: J_new = [I, [r]x; 0, I] @ J_old
    """
    X = np.eye(6)
    X[:3,3:] =  -skew(r)  
    return X

def jac_site_world_vw(model, data, site_id):
    """
    MuJoCo gives Jp (linear) and Jr (angular), both expressed in world.
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

# -------- textbook geometric Jacobian at an arbitrary world point ---------

def geometric_jacobian_world_at_point(model, data, p_ref_w, joint_indices=None):
    """
    Build the 6×n geometric Jacobian in the WORLD frame (v–ω ordering),
    evaluated at the reference point p_ref_w (world coords).

    Columns follow textbook formulas using:
      - revolute i:   J_v = z_i × (p_ref - p_i),  J_ω = z_i
      - prismatic i:  J_v = z_i,                  J_ω = 0

    z_i^w and p_i^w are obtained from MuJoCo joint frames:
      p_i^w = xpos(body_i) + R_body_i * jnt_pos_local
      z_i^w = R_body_i * jnt_axis_local
    """
    nv = model.nv
    if joint_indices is None:
        # assume first nv joints are the serial arm joints you care about
        joint_indices = list(range(nv))

    Jp = np.zeros((3, nv))
    Jr = np.zeros((3, nv))

    for col in joint_indices:
        jtype = model.jnt_type[col]           # mujoco.mjtJoint.mjJNT_HINGE or SLIDE
        jbid  = model.jnt_bodyid[col]         # body that owns this joint

        # body pose in world
        R_w_b = rot_from_xmat(data.xmat[jbid])
        p_w_b = data.xpos[jbid]

        # joint frame (position and axis) given in the body frame
        p_b_j = model.jnt_pos[col]            # 3-vector in body frame
        z_b_j = model.jnt_axis[col]           # 3-vector in body frame

        # transform to world
        p_w_j = p_w_b + R_w_b @ p_b_j
        z_w_j = R_w_b @ z_b_j

        if jtype == mujoco.mjtJoint.mjJNT_HINGE:      # revolute
            Jr[:, col] = z_w_j
            Jp[:, col] = np.cross(z_w_j, (p_ref_w - p_w_j))
        elif jtype == mujoco.mjtJoint.mjJNT_SLIDE:    # prismatic
            Jr[:, col] = np.zeros(3)
            Jp[:, col] = z_w_j
        else:
            # ignore other joint types (ball/free) or set zero
            Jr[:, col] = 0.0
            Jp[:, col] = 0.0

    return np.vstack([Jp, Jr])  # v–ω ordering

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

    # --- 5) Jacobian from API at the tool (reference) ---
    J_w_t_api = jac_site_world_vw(model, data, tool_site_id)        # v–ω
    print("\n=== J^{w}_{b->t} (API, v–ω) ===\n", J_w_t_api)

    # --- 6) Geometric Jacobian at the tool point (direct, textbook columns) ---
    # Use the first nv joints (typical serial arm); adjust if needed.
    J_w_t_geo_direct = geometric_jacobian_world_at_point(model, data, p_w_t)
    print("\n=== J^{w}_{b->t} (GEOMETRIC DIRECT at tool, v–ω) ===\n", J_w_t_geo_direct)

    # --- 7) Geometric Jacobian at the end-effector point, then point-shift to tool ---
    J_w_ee_geo = geometric_jacobian_world_at_point(model, data, p_w_ee)
    X_w_t_from_ee = X_point_shift_vw(r_w_t_ee)                      # [I, -[r]x; 0, I]
    J_w_t_from_ee = X_w_t_from_ee @ J_w_ee_geo
    print("\n=== J^{w}_{b->t} (EE GEOMETRIC, pre-multiplied by [I, -[r]x; 0, I]) ===\n", J_w_t_from_ee)

    # --- 8) Compare all three ---
    print("\nMax |API - GEOM_DIRECT|:", np.max(np.abs(J_w_t_api - J_w_t_geo_direct)))
    print("Max |API - (EE->TOOL via shift)|:", np.max(np.abs(J_w_t_api - J_w_t_from_ee)))
    print("Max |GEOM_DIRECT - (EE->TOOL via shift)|:", np.max(np.abs(J_w_t_geo_direct - J_w_t_from_ee)))

    # --- 9) Example torque from wrench at tool in world (v–ω) ---
    F_w_tool = np.array([0.0, 0.0, -20.0,   0.0, 0.0, 0.5])  # [force; moment]
    tau = J_w_t_api.T @ F_w_tool
    print("\nJoint torques from wrench at tool (world, v–ω):\n", tau)

    # (optional) collisions
    collisions = get_collisions(model, data, True)
    input("\nPress Enter to continue...")

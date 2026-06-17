import mujoco
import numpy as np
from transformations import rotm_to_quaternion, quaternion_to_euler, rotm2euler

def get_cartesian_pose(frame_id, data, representation):
    position = data.xpos[frame_id]
    rotation_matrix = data.xmat[frame_id].reshape(3, 3)
    quaternion = rotm_to_quaternion(rotation_matrix)
    euler_angles = quaternion_to_euler(quaternion, degrees=True)
    if representation == "quaternion":
        return position, quaternion
    elif representation == "euler":
        return position, euler_angles
    elif representation == "rotation_matrix":
        return position, rotation_matrix

def set_body_pose(model, data, body_id, pos, quat):
    model.body_pos[body_id] = pos
    model.body_quat[body_id] = quat
    mujoco.mj_forward(model, data)

def compute_jacobian(model, data, rob_par, tool_site_id):
    Jp = np.zeros((3, rob_par.nu))
    Jr = np.zeros((3, rob_par.nu))
    mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
    Jac = np.vstack([Jp, Jr])[:, :rob_par.nu]
    return Jac

def get_collisions(model, data, verbose):
    # Step the simulator once so that contacts get populated
    mujoco.mj_forward(model, data)

    if data.ncon == 0:
        if verbose: print("No collisions detected.")
    else:
        if verbose: print(f"{data.ncon} collision(s) detected:")
        for i in range(data.ncon):
            c = data.contact[i]
            # lookup names via mj_id2name
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            if verbose: print(f" - {name1} <-> {name2}")
            if verbose: print(f"Penetration depth: {c.dist}")
    return data.ncon

def inverse_manipulability(q, model, data, rob_par, tool_site_id):
    data.qpos[:rob_par.nu] = q 
    mujoco.mj_forward(model, data)
    J = compute_jacobian(model, data, rob_par, tool_site_id)
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return 1e12 if det <= 1e-6 else 1.0/np.sqrt(det)

def setup_target_frames(model, data, ref_body_ids, target_poses):
    for i, (pos, quat) in enumerate(target_poses):
        set_body_pose(model, data, ref_body_ids[i],
                      pos, [quat[3], quat[0], quat[1], quat[2]])
    mujoco.mj_forward(model, data)

def joint_displacement(q1, q2):
    """Euclidean distance accounting for joint angle wrapping."""
    diff = q1 - q2
    diff = (diff + np.pi) % (2 * np.pi) - np.pi 
    return np.linalg.norm(diff)





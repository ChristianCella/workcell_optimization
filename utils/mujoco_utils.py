import mujoco
import numpy as np

def set_body_pose(model, data, body_id, pos, quat):
    model.body_pos[body_id] = pos
    model.body_quat[body_id] = quat
    mujoco.mj_forward(model, data)

def compute_jacobian(model, data, tool_site_id):
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
    Jac = np.vstack([Jp, Jr])[:, :6]
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
            if verbose: print(f"  • {name1} ↔ {name2}")
    return data.ncon

def inverse_manipulability(q, model, data, tool_site_id):
    data.qpos[:model.nv] = q; mujoco.mj_forward(model, data)
    Jp = np.zeros((3,model.nv)); Jr = np.zeros((3,model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
    J = np.vstack([Jp, Jr])[:,:model.nv]
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return 1e12 if det <= 1e-12 else 1.0/np.sqrt(det)

def directional_inverse_manipulability(q, model, data, tool_site_id, u_z):
    data.qpos[:model.nv] = q; mujoco.mj_forward(model, data)
    Jp = np.zeros((3,model.nv)); Jr = np.zeros((3,model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
    J = np.vstack([Jp, Jr])[:,:6]
    dir_inv_man = u_z.T @ J @ J.T @ u_z
    return 1e12 if dir_inv_man <= 1e-12 else 1.0/np.sqrt(dir_inv_man)

def setup_target_frames(model, data, ref_body_ids, target_poses):
    for i, (pos, quat) in enumerate(target_poses):
        set_body_pose(model, data, ref_body_ids[i],
                      pos, [quat[3], quat[0], quat[1], quat[2]])
    mujoco.mj_forward(model, data)


#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time, os, sys
import threading
import concurrent.futures

# ---------- Your Original Helper Functions ----------

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def set_body_pose(model, data, body_id, pos, quat):
    model.body_pos[body_id] = pos
    model.body_quat[body_id] = quat
    mujoco.mj_forward(model, data)

def get_tool_z_direction(data, tool_site_id):
    mat = data.site_xmat[tool_site_id].reshape(3, 3)
    return mat[:, 2]

def five_dof_error(q, model, data, tool_site_id, target_pos, z_dir_des):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[tool_site_id]
    z_dir = get_tool_z_direction(data, tool_site_id)
    pos_err = target_pos - pos
    err_axis = np.cross(z_dir, z_dir_des)
    err_proj = err_axis - np.dot(err_axis, z_dir_des) * z_dir_des
    return np.concatenate([pos_err, err_proj[:2]])

def five_dof_error_with_sign(q, model, data, tool_site_id, target_pos, z_dir_des):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[tool_site_id]
    z_dir = get_tool_z_direction(data, tool_site_id)
    pos_err = target_pos - pos
    err_axis = np.cross(z_dir, z_dir_des)
    err_proj = err_axis - np.dot(err_axis, z_dir_des) * z_dir_des
    sign_penalty = max(0, -np.dot(z_dir, z_dir_des))  # Only positive when dot < 0
    return np.concatenate([pos_err, err_proj[:2], [sign_penalty]])

def get_full_jacobian(model, data, tool_site_id):
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id)
    J = np.vstack([jacp, jacr])
    return J[:, :6]

def manipulability(q, model, data, tool_site_id):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    J = get_full_jacobian(model, data, tool_site_id)
    JJt = J @ J.T
    if np.linalg.matrix_rank(JJt) < 6 or np.linalg.det(JJt) < 1e-12:
        return 1e6  # Return large value if near singularity (for inverse manipulability)
    return -(np.sqrt(np.linalg.det(JJt)))

def geom_collision_penalty(q, model, data, robot_geom_ids, floor_geom_id=None, collision_weight=1e5):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    penalty = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        robot_vs_robot = (g1 in robot_geom_ids and g2 in robot_geom_ids and g1 != g2)
        robot_vs_floor = (floor_geom_id is not None) and (
            (g1 in robot_geom_ids and g2 == floor_geom_id) or (g2 in robot_geom_ids and g1 == floor_geom_id)
        )
        if robot_vs_robot or robot_vs_floor:
            penetration = max(0, -c.dist)
            penalty += 1.0 + 1000.0 * penetration  # Strong penalty for penetration
    return collision_weight * penalty

# Cost weights
W_POSE = 1e10
W_JOINT_DISP = 0
W_LIMITS = 0 # 1e2
W_ELBOW = 0
W_MANIP = 1e2 # 1e4
W_ANTIALIGN = 1e3

def bioik_cost(q, model, data, tool_site_id, target_pos, z_dir_des, q_seed, joint_lims, elbow_pref=None,
               robot_geom_ids=None, floor_geom_id=None, collision_weight=1e5):
    pose_err = five_dof_error(q, model, data, tool_site_id, target_pos, z_dir_des)
    cost_pose = np.sum(pose_err**2)
    cost_joint_disp = np.sum((q - q_seed)**2)
    lower, upper = joint_lims[:,0], joint_lims[:,1]
    mid = 0.5 * (lower + upper)
    range_ = upper - lower
    cost_limits = np.sum(((2 * np.abs(q - mid) - 0.5 * range_)**2))
    if elbow_pref is not None:
        el, eh = elbow_pref
        cost_elbow = (2 * np.abs(q[3] - 0.5*(eh + el)) - 0.5*(eh - el))**2
    else:
        cost_elbow = 0.0
    manip = manipulability(q, model, data, tool_site_id)
    collision_penalty = 0.0
    if robot_geom_ids is not None:
        collision_penalty = geom_collision_penalty(q, model, data, robot_geom_ids, floor_geom_id, collision_weight=collision_weight)
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    z_dir = get_tool_z_direction(data, tool_site_id)
    dot_z = np.dot(z_dir, z_dir_des)
    anti_align_penalty = 0.0
    if dot_z < 0:
        anti_align_penalty = -dot_z
    cost = (W_POSE * cost_pose +
            W_JOINT_DISP * cost_joint_disp +
            W_LIMITS * cost_limits +
            W_ELBOW * cost_elbow +
            W_MANIP * manip +
            collision_penalty +
            W_ANTIALIGN * anti_align_penalty)
    return cost

# -------------- Fast, Parallelized Multi-Seed Local IK --------------

def fast_parallel_ik(model, data, tool_site_id, target_pos, z_dir_des, joint_lims,
                     robot_geom_ids, floor_geom_id, elbow_pref, 
                     n_seeds=16, n_local=4, local_maxiter=40, verbose=False):
    """ 
    Parallelized, multi-seed local IK, returns best solution in <0.1s 
    """
    np.random.seed(42)
    seeds = [np.random.uniform(joint_lims[:,0], joint_lims[:,1]) for _ in range(n_seeds)]
    seeds[0] = np.zeros(6)

    def cost_wrap(q, q_seed):
        # Make a new MjData for thread safety
        thread_data = mujoco.MjData(model)
        mujoco.mj_resetData(model, thread_data)
        return bioik_cost(q, model, thread_data, tool_site_id, target_pos, z_dir_des, q_seed=q_seed,
                          joint_lims=joint_lims, elbow_pref=elbow_pref,
                          robot_geom_ids=robot_geom_ids, floor_geom_id=floor_geom_id, collision_weight=1e5)

    # Batch evaluate seeds in parallel
    costs = [None]*len(seeds)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_seeds, os.cpu_count())) as executor:
        futures = [executor.submit(cost_wrap, q, q) for q in seeds]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            costs[i] = f.result()
    costs = np.array(costs)
    best_idxs = np.argsort(costs)[:n_local]
    best_qs = [seeds[i] for i in best_idxs]

    def local_opt(q0):
        thread_data = mujoco.MjData(model)
        mujoco.mj_resetData(model, thread_data)
        res = minimize(lambda q: cost_wrap(q, q0),
                       q0,
                       method='L-BFGS-B',
                       bounds=list(zip(joint_lims[:,0], joint_lims[:,1])),
                       options={'ftol': 1e-6, 'maxiter': local_maxiter, 'disp': False})
        return res.x, res.fun

    local_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_local) as executor:
        futures = [executor.submit(local_opt, q0) for q0 in best_qs]
        for f in concurrent.futures.as_completed(futures):
            local_results.append(f.result())
    local_results = sorted(local_results, key=lambda x: x[1])
    q_best, cost_best = local_results[0]
    if verbose:
        print(f"Best local cost: {cost_best}")
    return q_best

# -------------- Main Routine --------------

def setup_gpu_context():
    print("Setting up GPU context...")
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            print(f"Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name} ({gpu.memoryTotal}MB)")
        else:
            print("No GPUs found")
    except ImportError:
        print("GPUtil not available, cannot check GPU status")
    os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless GPU rendering
    print("GPU context setup complete")

def main():
    verbose = True
    show_pose_duration = 5.0
    setup_gpu_context()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, "universal_robots_ur5e", "scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1080
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    ref_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reference_target")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")

    excluded_prefixes = (
        "floor", "world_", "reference_", "force_arrow", "moment_arrow"
    )
    robot_geom_ids = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        body_id = model.geom_bodyid[i]
        if name is not None:
            if any(name.startswith(prefix) for prefix in excluded_prefixes):
                continue
        if body_id <= 0:
            continue
        is_collision = False
        if name is None:
            is_collision = True
        elif "collision" in name or "eef_collision" in name:
            is_collision = True
        if is_collision:
            robot_geom_ids.append(i)
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    print("Robot geom ids:", robot_geom_ids)
    print("Floor geom id:", floor_geom_id)

    print("\nGeoms list:")
    for i in range(model.ngeom):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))

    target_poses = [
        (np.array([0.2, 0.2, 0.1]), R.from_euler('xyz', [180, 37, 0], degrees=True).as_quat()),
    ]

    joint_lims = model.jnt_range[:6]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start the pose...")

        for pos, quat in target_poses:
            model.body_pos[base_body_id] = [-0.1, -0.1, 0.2]
            model.body_quat[base_body_id] = euler_to_quaternion(45, 0, 0, degrees=True)
            model.body_pos[tool_body_id] = [0.1, 0.1, 0.25]
            model.body_quat[tool_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)

            set_body_pose(model, data, ref_body_id, pos, [quat[3], quat[0], quat[1], quat[2]])
            mujoco.mj_forward(model, data)

            R_target = R.from_quat(quat).as_matrix()
            z_dir_des = R_target[:, 2].copy()
            elbow_pref = (joint_lims[3,0], joint_lims[3,0]+0.4*(joint_lims[3,1]-joint_lims[3,0]))

            print("\nFast multi-seed parallel IK...")
            start_time = time.time()

            q_opt = fast_parallel_ik(model, data, tool_site_id, pos, z_dir_des, joint_lims,
                                     robot_geom_ids, floor_geom_id, elbow_pref,
                                     n_seeds=80, n_local=32, local_maxiter=500, verbose=verbose)

            solve_time = time.time() - start_time
            print(f"IK optimization completed in {solve_time:.4f} seconds")

            data.qpos[:6] = q_opt
            mujoco.mj_forward(model, data)
            print("\nFinal configuration:", np.round(q_opt, 3))
            print("Tool site:", np.round(data.site_xpos[tool_site_id],3))
            print("z_dir:", np.round(get_tool_z_direction(data, tool_site_id),3))
            print("Pose cost (should be ~0):", np.sum(five_dof_error_with_sign(q_opt, model, data, tool_site_id, pos, z_dir_des)**2))
            print("Inverse manipulability:", np.abs(1/manipulability(q_opt, model, data, tool_site_id)))
            print(f"Total optimization time: {solve_time:.4f} seconds")

            viewer.sync()
            time.sleep(show_pose_duration)

        if verbose: print("\n--- Finished ---")
        input("Press Enter to close the viewer...")

if __name__ == "__main__":
    main()

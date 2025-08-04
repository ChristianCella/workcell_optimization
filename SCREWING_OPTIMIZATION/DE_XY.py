#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution, minimize
import time, os, sys
import threading


def torque_value( model, data, tool_site_id, z_dir_des):
    external_force_world = np.array([0, 0.0, 10.0])
    external_torque_world = np.array([0.0, 0.0, 0.0])
    wrench = np.hstack([external_force_world, external_torque_world]) 
    J = get_full_jacobian(model, data, tool_site_id)
    tau = J.T@ wrench
    obj_func = np.sqrt(tau.T @ tau)
    return obj_func

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
    """
    Penalizes self-collisions and collisions with the floor using MuJoCo's geom contacts.
    """
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
W_POSE = 1e10 # 1e12
W_JOINT_DISP = 0
W_LIMITS = 0 # 1e2
W_ELBOW = 0
W_MANIP = 1e3 # 1e4
W_ANTIALIGN = 1e4 # 1e3


def pref_manipulability(q, model, data, tool_site_id,z_dir_des):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    J = get_full_jacobian(model, data, tool_site_id)
    f=np.append(z_dir_des, np.zeros(3))
    JJt = J @ J.T
    JJt_pref=f.T @ JJt @ f
    if np.linalg.matrix_rank(JJt) < 6 or np.linalg.det(JJt) < 1e-12:
        return 1e6  # Return large value if near singularity (for inverse manipulability)
    return (1/JJt_pref)


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

def setup_target_frames(model, data, ref_body_ids, target_poses):
    """Setup all target frames at their specified poses"""
    for i, (pos, quat) in enumerate(target_poses):
        set_body_pose(model, data, ref_body_ids[i], pos, [quat[3], quat[0], quat[1], quat[2]])
    mujoco.mj_forward(model, data)

def func(x,y):
    total_tau = 0

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, "universal_robots_ur5e", "scene.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.vis.global_.offwidth = 0
    model.vis.global_.offheight = 0

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)   
    
    # Get reference body IDs for multiple target frames
    ref_body_ids = []
    for i in range(1):  # Assuming you have N reference targets
        try:
            ref_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"reference_target_{i+1}")
            if ref_body_id == -1:
                # Fallback to single reference target if multiple don't exist
                ref_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reference_target")
            ref_body_ids.append(ref_body_id)
        except:
            # If reference_target_{i+1} doesn't exist, use the main reference_target
            ref_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reference_target")
            ref_body_ids.append(ref_body_id)
    
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")

    excluded_prefixes = ("floor", "world_", "reference_", "force_arrow", "moment_arrow")
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

   

    # Define N target poses for the video
    target_poses = [
        (np.array([0.2, -0.2, 0.2]), R.from_euler('XYZ', [180, 0, 0], degrees=True).as_quat()),
    ]
    
    joint_lims = model.jnt_range[:6]
    q_seed = np.zeros(6)  # Seed for optimization
    q_start = np.radians([-8.38, -68.05, -138, -64, 90, -7.85])

    # Set base and tool body poses
    model.body_pos[base_body_id] = [x, y, 0]
    model.body_quat[base_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)
    model.body_pos[tool_body_id] = [0.1, 0.1, 0.25]
    model.body_quat[tool_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)
        
    # Set robot to starting configuration
    data.qpos[:6] = q_start
    data.qvel[:] = 0  # clear velocities
    data.qacc[:] = 0  # clear accelerations
    data.ctrl[:] = 0  # (if using actuators, may help avoid torque pollution)
        
    # Setup all target frames
    setup_target_frames(model, data, ref_body_ids, target_poses)
        
    # Update visualization
    mujoco.mj_forward(model, data)
        
    # Optimize for each target frame
    for target_idx, (pos, quat) in enumerate(target_poses):
                     
        R_target = R.from_quat(quat).as_matrix()
        z_dir_des = R_target[:, 2].copy()
        elbow_pref = (joint_lims[3,0], joint_lims[3,0]+0.4*(joint_lims[3,1]-joint_lims[3,0]))

        # STart computing time
        start_time = time.time()

        def cost_wrap(q):
            return bioik_cost(q, model, data, tool_site_id, pos, z_dir_des, q_seed=q_seed,
                                joint_lims=joint_lims, elbow_pref=elbow_pref,
                                robot_geom_ids=robot_geom_ids, floor_geom_id=floor_geom_id, collision_weight=1e5)

        bounds = list(zip(joint_lims[:, 0], joint_lims[:, 1]))

        # Global optimization
        result = differential_evolution(
            cost_wrap,
            bounds,
            popsize = 30,
            maxiter = 250,
            polish = False,
            workers = 1,
            updating = 'immediate',
            seed = 42
        )
        q_global = result.x
        global_time = time.time() - start_time
        print(f"Global optimization took {global_time:.2f} seconds for target {target_idx + 1}")
        
        # Update robot configuration and display
        data.qpos[:6] = q_global
        mujoco.mj_forward(model, data)
        

        # Local refinement      
        start_time = time.time()
        def local_cost(q):
            return bioik_cost(q, model, data, tool_site_id, pos, z_dir_des, q_seed=q_global,
                                joint_lims=joint_lims, elbow_pref=elbow_pref,
                                robot_geom_ids=robot_geom_ids, floor_geom_id=floor_geom_id, collision_weight=1e5)

        res = minimize(
            local_cost,
            q_global,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-8, 'maxiter': 100, 'maxfun': 500}
        )
        q_opt = res.x
        local_time = time.time() - start_time
        print(f"Local optimization took {local_time:.2f} seconds for target {target_idx + 1}")
        
        # Update robot to final optimized configuration
        data.qpos[:6] = q_opt
        mujoco.mj_forward(model, data)
        
        # Display results      
        total_tau = total_tau + torque_value(model, data, tool_site_id, z_dir_des)    

    return total_tau   

if __name__ == "__main__":
    # Example usage
    x = 0.2
    y = -0.2
    result = func(x, y)

    print(f"Result for position ({x}, {y}): {result}")
    
import sys, time, os, mujoco, torch
from mujoco_utils import *
from transformations import *   
from custom_ik_solvers import *
import fonts
from scipy.signal import savgol_filter

#* Base directrory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

#* Directory for scene creation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from config import *
rob_params = rob_par
from parameters import TestIK
ik_params = TestIK()

#* Directory for the utilities
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
import fonts
from transformations import rotm_to_quaternion, rotm2euler, get_homogeneous_matrix
from mujoco_utils import set_body_pose
from ikflow_inference import FastIKFlowSolver, solve_ik_fast
fast_ik_solver = FastIKFlowSolver()

def create_path(cartesian_path, model, data, rob_par, tool_tip_site_id, A_w_b, A_ee_t, A_wl3_ee, viewer):
    start_time = time.time()
    q_path = []
    cols   = []
    reach  = []

    n = len(cartesian_path) #* number of waypoints
    q_home = data.qpos[:rob_par.nu].copy() #* home configuration 

    #! Use ikflow on the first waypoint
    t_w_p = cartesian_path[0][0]
    theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 = cartesian_path[0][1]

    sols_ok, fk_ok = [], []
    for i in range(ik_params.N_disc):
        R_w_p_rotated = R.from_euler('XYZ',[theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 + i * 2 * np.pi / ik_params.N_disc], degrees=False).as_matrix()
        A_w_p_rotated = np.eye(4)
        A_w_p_rotated[:3, 3] = t_w_p
        A_w_p_rotated[:3, :3] = R_w_p_rotated

        A_b_wl3 = (np.linalg.inv(A_w_b) @ A_w_p_rotated @ np.linalg.inv(A_ee_t) @ np.linalg.inv(A_wl3_ee))
        quat_pose = rotm_to_quaternion(A_b_wl3[:3, :3])
        target = np.array([A_b_wl3[0, 3], A_b_wl3[1, 3], A_b_wl3[2, 3], quat_pose[0],  quat_pose[1],   quat_pose[2], quat_pose[3]], dtype=np.float64)

        tgt_tensor = torch.from_numpy(target.astype(np.float32))
        sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N=ik_params.N_samples, fast_solver=fast_ik_solver)
        sols_ok.append(sols_disc)
        fk_ok.append(fk_disc)

    sols_np = torch.cat(sols_ok, dim=0).cpu().numpy()

    #* it is not possible to find a ik for the first waypoint
    if sols_np.shape[0] == 0:
        print(f"{fonts.red}IKFlow found no solutions for waypoint 0{fonts.reset}")
        total_time = time.time() - start_time

        # path, reach, cols, total_time
        return [], [1] * n, [0] * n, total_time

    #! At least one ik solution for waypoint 1
    free_mask = np.zeros(len(sols_np), dtype=bool)
    for i, q in enumerate(sols_np):
        data.qpos[:rob_par.nu] = q.tolist()
        mujoco.mj_forward(model, data)
        if verbose:
            viewer.sync()
            print(f"Checking collision for seed {i + 1}/{len(sols_np)}: {q}")
            input(f"Press Enter to continue...")
        if get_collisions(model, data, False) == 0:
            free_mask[i] = True

    free_sols = sols_np[free_mask] #* collision-free solutions

    if free_sols.shape[0] == 0:
        print(f"{fonts.red}No collision-free IKFlow seeds found for waypoint 0{fonts.reset}")
        total_time = time.time() - start_time

        # path, reach, cols, total_time
        return [], [1] * n, [0] * n, total_time

    #! Among the many col-free solutions, get the closer to q_home
    distances = np.sum((free_sols - q_home[np.newaxis, :]) ** 2, axis=1)
    ranked_indices = np.argsort(distances)
    ranked_seeds = free_sols[ranked_indices] #* ranked version of free_sols

    print(f"{fonts.blue}Waypoint 0: {free_sols.shape[0]} collision-free seeds, "
            f"closest to q_home: {np.sqrt(distances[ranked_indices[0]]):.4f} rad{fonts.reset}")

    #! Now propagate ik solutions through an ik solver
    #* supported methods: damped least squares (dls) or model-based inverse kinemeatics (mink)
    success = False
    candidate_path = []
    candidate_reach = []
    candidate_cols = []

    for seed_idx, q_seed in enumerate(ranked_seeds):
        candidate_path = [q_seed]
        candidate_reach = [0]
        candidate_cols = [0]
        collision_found = False

        data.qpos[:rob_par.nu] = q_seed.tolist()
        mujoco.mj_forward(model, data)
        if verbose:
            viewer.sync()
            print(f"Initial q: {q_seed}")
            #input(f"press enter")

        #* Try to exapnd this seed till the end
        q_prev = q_seed.copy()
        for j in range(1, n):
            if verbose: print(f"Waypoint {j}")
            t_w_p = cartesian_path[j][0]
            theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 = cartesian_path[j][1]

            if ik_solver_to_use == "mink":
                target_rot = R.from_euler('xyz', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0], degrees=False).as_matrix()
                data.qpos[:rob_par.nu] = q_prev.tolist()
                mujoco.mj_forward(model, data)
                if verbose:
                    print(f"Current q: {q_prev}")
                    viewer.sync()
                    #input(f"Press Enter to continue to waypoint {j}...")
                #! res = 0 if the new config is close to the old one
                best_q, res = ik_mink("tool_frame", model, t_w_p, target_rot, q_init=q_prev)
            elif ik_solver_to_use == "dls":
                target_rot = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0], degrees=False).as_matrix()
                data.qpos[:rob_par.nu] = q_prev.tolist()
                mujoco.mj_forward(model, data)
                if verbose:
                    print(f"Current q: {q_prev}")                  
                    viewer.sync()
                    #input(f"Press Enter to continue to waypoint {j}...")               
                #! res = 0 if the error is too big
                best_q, res = solve_ik_dls(model, data, rob_par, tool_tip_site_id, t_w_p, target_rot, q_init=q_prev, max_iter=200, tol=1e-5, lam_max=0.1, eps=1e-6, dq_max=0.05)
            elif ik_solver_to_use == "trackik":
                target_rot = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0], degrees=False).as_matrix()
                data.qpos[:rob_par.nu] = q_prev.tolist()
                mujoco.mj_forward(model, data)
                if verbose:
                    print(f"Current q: {q_prev}")
                    viewer.sync()
                    #input(f"Press Enter to continue to waypoint {j}...")               
                best_q, res = solve_ik_tracik(model, data, rob_par, tool_tip_site_id, t_w_p, target_rot, q_init=q_prev)
            else:
                raise ValueError(f"Unknown IK solver: {ik_solver_to_use}")
                           
            #! IK failure
            if res != 0:
                if verbose: 
                    print(f"{fonts.red}The IK solver {ik_solver_to_use} failed at waypoint {j}!{fonts.reset}")
                    data.qpos[:rob_par.nu] = best_q.tolist()
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    J = compute_jacobian(model, data, rob_par, tool_tip_site_id)
                    print(f"det(J)={np.linalg.det(J):.4e}")
                    cols = get_collisions(model, data, True)
                    input(f"Press Enter to continue to the next seed...")
                candidate_reach.append(1)
                candidate_cols.append(0)
                candidate_path.append(best_q)
                collision_found = True
                break

            #* Collision check
            data.qpos[:rob_par.nu] = best_q.tolist()
            mujoco.mj_forward(model, data)
            if verbose: viewer.sync()
            n_cols = get_collisions(model, data, verbose)

            if n_cols > 0:
                print(f"{fonts.red}Collision detected at waypoint {j}!{fonts.reset}")
                if verbose: 
                    viewer.sync()
                    input(f"Press Enter to continue to the next seed...")
                collision_found = True
                
                break

            #* IK found, point is reachable and no collision: add to candidate path
            candidate_path.append(best_q)
            candidate_reach.append(0)
            candidate_cols.append(0)
            q_prev = best_q.copy()

        #! If we reached the end of the path without collisions, we are good
        if not collision_found:
            q_path = candidate_path
            reach = candidate_reach
            cols = candidate_cols
            success = True
            print(f"{fonts.green}Path found using seed {seed_idx + 1}/{len(ranked_seeds)}{fonts.reset}")
            break
        else:
            print(f"{fonts.yellow}Seed {seed_idx + 1}/{len(ranked_seeds)} failed "
                    f"— trying next{fonts.reset}")
    #! At this point, eitehr we found a path or we failed
    if not success:
        print(f"{fonts.red}No fully collision-free path found after trying "
                f"all {len(ranked_seeds)} seeds{fonts.reset}")
        q_path = candidate_path
        reach = candidate_reach
        cols = candidate_cols
        missing = n - len(q_path)
        if missing > 0:
            q_path += [q_path[-1]] * missing #* Fake path where we copy the last configuration until the end
            reach += [1] * missing
            cols += [0] * missing

    # Keep MuJoCo state consistent with final path
    data.qpos[:rob_par.nu] = q_path[-1].tolist()
    mujoco.mj_forward(model, data)
    
    #! Post-processing
    q_path = np.unwrap(q_path, axis=0)
    total_time = time.time() - start_time

    return q_path, reach, cols, total_time


def smooth_q_path(q_path, window=5, polyorder=3):
    """
    Apply Savitzky-Golay filter to each joint independently.
    window : must be odd and > polyorder
    polyorder: polynomial order for the filter
    """
    q_path = np.asarray(q_path)
    q_smooth = np.zeros_like(q_path)
    for j in range(q_path.shape[1]):
        q_smooth[:, j] = savgol_filter(q_path[:, j], window_length=window, polyorder=polyorder)
    return q_smooth
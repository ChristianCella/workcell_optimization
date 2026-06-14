import sys, time, os, mujoco, torch
from mujoco_utils import *
from transformations import *   
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
fast_ik_solver = None
if ik_solver_to_use == "ikflow": fast_ik_solver = FastIKFlowSolver()

def create_path(cartesian_path, model, data, rob_par, tool_tip_site_id, A_w_b, A_ee_t, A_wl3_ee, save_data):
    start_time = time.time()
    q_path = []
    cols   = []
    reach  = []

    # ------------------------------------------------------------------ #
    # IKFlow branch                                                        #
    # ------------------------------------------------------------------ #
    if ik_solver_to_use == "ikflow":

        n = len(cartesian_path)
        q_home = data.qpos[:rob_par.nu].copy()

        # ------------------------------------------------------------------ #
        # PHASE 1 — use IKFlow only on the FIRST waypoint                    #
        # ------------------------------------------------------------------ #
        t_w_p = cartesian_path[0][0]
        theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 = cartesian_path[0][1]

        sols_ok, fk_ok = [], []
        for i in range(ik_params.N_disc):
            R_w_p_rotated = R.from_euler(
                'XYZ',
                [theta_w_p_x_0,
                 theta_w_p_y_0,
                 theta_w_p_z_0 + i * 2 * np.pi / ik_params.N_disc],
                degrees=False
            ).as_matrix()

            A_w_p_rotated         = np.eye(4)
            A_w_p_rotated[:3, 3]  = t_w_p
            A_w_p_rotated[:3, :3] = R_w_p_rotated

            A_b_wl3 = (
                np.linalg.inv(A_w_b)
                @ A_w_p_rotated
                @ np.linalg.inv(A_ee_t)
                @ np.linalg.inv(A_wl3_ee)
            )
            quat_pose = rotm_to_quaternion(A_b_wl3[:3, :3])
            target = np.array([
                A_b_wl3[0, 3], A_b_wl3[1, 3], A_b_wl3[2, 3],
                quat_pose[0],  quat_pose[1],   quat_pose[2], quat_pose[3]
            ], dtype=np.float64)

            tgt_tensor         = torch.from_numpy(target.astype(np.float32))
            sols_disc, fk_disc = solve_ik_fast(
                tgt_tensor, N=ik_params.N_samples, fast_solver=fast_ik_solver
            )
            sols_ok.append(sols_disc)
            fk_ok.append(fk_disc)

        sols_np = torch.cat(sols_ok, dim=0).cpu().numpy()

        if sols_np.shape[0] == 0:
            print(f"{fonts.red}IKFlow found no solutions for waypoint 0{fonts.reset}")
            total_time = time.time() - start_time
            return [], [1] * n, [0] * n, total_time

        # ------------------------------------------------------------------ #
        # Filter collision-free seeds                                         #
        # ------------------------------------------------------------------ #
        free_mask = np.zeros(len(sols_np), dtype=bool)

        for i, q in enumerate(sols_np):
            data.qpos[:rob_par.nu] = q.tolist()
            mujoco.mj_forward(model, data)
            if get_collisions(model, data, False) == 0:
                free_mask[i] = True

        free_sols = sols_np[free_mask]

        if free_sols.shape[0] == 0:
            print(f"{fonts.red}No collision-free IKFlow seeds found for waypoint 0{fonts.reset}")
            total_time = time.time() - start_time
            return [], [1] * n, [0] * n, total_time

        # Sort by L2 distance from q_home (ascending — closest first)
        distances      = np.sum((free_sols - q_home[np.newaxis, :]) ** 2, axis=1)
        ranked_indices = np.argsort(distances)
        ranked_seeds   = free_sols[ranked_indices]

        print(f"{fonts.blue}Waypoint 0: {free_sols.shape[0]} collision-free seeds, "
              f"closest to q_home: {np.sqrt(distances[ranked_indices[0]]):.4f} rad{fonts.reset}")

        # ------------------------------------------------------------------ #
        # PHASE 2 — for each seed, run DLS forward along the full path       #
        # Tight dq_max prevents branch switching                              #
        # ------------------------------------------------------------------ #
        success = False
        candidate_path  = []
        candidate_reach = []
        candidate_cols  = []

        for seed_idx, q_seed in enumerate(ranked_seeds):

            candidate_path  = [q_seed]
            candidate_reach = [0]
            candidate_cols  = [0]
            collision_found = False

            q_prev = q_seed.copy()

            for j in range(1, n):
                t_w_p = cartesian_path[j][0]
                theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 = cartesian_path[j][1]

                target_rot = R.from_euler(
                    'XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0],
                    degrees=False
                ).as_matrix()

                data.qpos[:rob_par.nu] = q_prev.tolist()
                mujoco.mj_forward(model, data)

                # Tight dq_max: correct solution is ~0.05 rad away,
                # wrong branch is ~1.86 rad away — cannot be reached
                # within max_iter=50 steps of 0.01 rad each
                best_q, res = solve_ik_dls(
                    model, data, rob_par, tool_tip_site_id,
                    t_w_p, target_rot, q_init=q_prev,
                    max_iter=200, tol=1e-5, lam_max=0.1, eps=1e-6, dq_max=0.05
                )

                # IK failure
                if res != 0:
                    candidate_reach.append(1)
                    candidate_cols.append(0)
                    candidate_path.append(best_q)
                    collision_found = True
                    break

                # Collision check
                data.qpos[:rob_par.nu] = best_q.tolist()
                mujoco.mj_forward(model, data)
                n_cols = get_collisions(model, data, False)

                if n_cols > 0:
                    collision_found = True
                    break

                candidate_path.append(best_q)
                candidate_reach.append(0)
                candidate_cols.append(0)
                q_prev = best_q.copy()

            if not collision_found:
                q_path = candidate_path
                reach  = candidate_reach
                cols   = candidate_cols
                success = True
                print(f"{fonts.green}Path found using seed {seed_idx + 1}/{len(ranked_seeds)}{fonts.reset}")
                break
            else:
                print(f"{fonts.yellow}Seed {seed_idx + 1}/{len(ranked_seeds)} failed "
                      f"— trying next{fonts.reset}")

        if not success:
            print(f"{fonts.red}No fully collision-free path found after trying "
                  f"all {len(ranked_seeds)} seeds{fonts.reset}")
            q_path = candidate_path
            reach  = candidate_reach
            cols   = candidate_cols
            missing = n - len(q_path)
            if missing > 0:
                q_path += [q_path[-1]] * missing
                reach  += [1] * missing
                cols   += [0] * missing

        # Keep MuJoCo state consistent with final path
        data.qpos[:rob_par.nu] = q_path[-1].tolist()
        mujoco.mj_forward(model, data)

        # ------------------------------------------------------------------ #
        # POST-PROCESSING — fix residual branch switches iteratively          #
        # Uses looser parameters than main pass                               #
        # ------------------------------------------------------------------ #
        JUMP_THRESHOLD = 0.3
        POSE_TOL       = 1e-3

        q_path_arr = np.array(q_path)
        changed = True
        passes  = 0

        while changed and passes < 5:
            changed = False
            passes += 1

            for j in range(1, len(q_path_arr) - 1):
                jump_left  = np.abs(q_path_arr[j] - q_path_arr[j - 1]).max()
                jump_right = np.abs(q_path_arr[j] - q_path_arr[j + 1]).max()

                if max(jump_left, jump_right) <= JUMP_THRESHOLD:
                    continue

                print(f"{fonts.yellow}Jump at waypoint {j} "
                      f"(L:{jump_left:.3f} R:{jump_right:.3f} rad on joint "
                      f"{np.abs(q_path_arr[j] - q_path_arr[j-1]).argmax()+1}) "
                      f"— re-solving (pass {passes}){fonts.reset}")

                t_w_p = cartesian_path[j][0]
                theta_x, theta_y, theta_z = cartesian_path[j][1]
                target_rot = R.from_euler(
                    'XYZ', [theta_x, theta_y, theta_z], degrees=False
                ).as_matrix()

                best_candidate  = None
                best_total_jump = max(jump_left, jump_right)

                seeds_to_try = [
                    q_path_arr[j - 1],
                    q_path_arr[j + 1],
                    (q_path_arr[j - 1] + q_path_arr[j + 1]) / 2.0
                ]

                for q_seed in seeds_to_try:
                    data.qpos[:rob_par.nu] = q_seed.tolist()
                    mujoco.mj_forward(model, data)

                    best_q, res = solve_ik_dls(
                        model, data, rob_par, tool_tip_site_id,
                        t_w_p, target_rot, q_init=q_seed,
                        max_iter=500, tol=1e-9, lam_max=0.5, eps=1e-8,
                        dq_max=0.05
                    )

                    # Check actual pose error rather than trusting res
                    data.qpos[:rob_par.nu] = best_q.tolist()
                    mujoco.mj_forward(model, data)

                    curr_pos = data.site_xpos[tool_tip_site_id]
                    curr_rot = data.site_xmat[tool_tip_site_id].reshape(3, 3)
                    R_err    = target_rot @ curr_rot.T
                    err_rot  = 0.5 * np.array([
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1]
                    ])
                    err      = np.concatenate([t_w_p - curr_pos, err_rot])
                    pose_err = np.linalg.norm(err)

                    if pose_err > POSE_TOL:
                        continue

                    if get_collisions(model, data, False) > 0:
                        continue

                    total_jump = max(
                        np.abs(best_q - q_path_arr[j - 1]).max(),
                        np.abs(best_q - q_path_arr[j + 1]).max()
                    )

                    if total_jump < best_total_jump:
                        best_total_jump = total_jump
                        best_candidate  = best_q

                if best_candidate is not None:
                    old_jump = max(jump_left, jump_right)
                    q_path_arr[j] = best_candidate
                    changed = True
                    print(f"{fonts.green}  Fixed: jump reduced from "
                          f"{old_jump:.4f} to {best_total_jump:.4f} rad{fonts.reset}")
                else:
                    print(f"{fonts.red}  Could not fix waypoint {j}{fonts.reset}")

        q_path = list(q_path_arr)

    # ------------------------------------------------------------------ #
    # DLS branch — untouched                                               #
    # ------------------------------------------------------------------ #
    elif ik_solver_to_use == "dls":
        for j in range(len(cartesian_path)):
            q_old = data.qpos[:rob_par.nu].copy()
            t_w_p = cartesian_path[j][0]
            theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 = cartesian_path[j][1]

            target_rot = R.from_euler(
                'XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0],
                degrees=False
            ).as_matrix()
            best_q, res = solve_ik_dls(
                model, data, rob_par, tool_tip_site_id,
                t_w_p, target_rot, q_init=q_old,
                max_iter=100, tol=1e-7, lam_max=0.1, eps=1e-6, dq_max=0.05
            )
            data.qpos[:rob_par.nu] = best_q
            mujoco.mj_forward(model, data)
            q_path.append(best_q)
            reach.append(res)

    else:
        raise ValueError(f"Unknown IK solver type: {ik_solver_to_use}")

    # ---------------------------------------------------------------------- #
    # Common post-processing                                                  #
    # ---------------------------------------------------------------------- #
    q_path     = np.unwrap(q_path, axis=0)
    total_time = time.time() - start_time

    if save_data:
        q_path_np = np.array(q_path)
        csv_path  = os.path.join(
            base_dir, "workcell_optimization/results",
            f"q_path_{robot_to_use}_{ik_solver_to_use}.csv"
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        header = ",".join([f"q{i+1}" for i in range(rob_par.nu)])
        np.savetxt(csv_path, q_path_np, delimiter=",",
                   header=header, comments="")
        print(f"{fonts.green}Path saved to {csv_path}{fonts.reset}")

    return q_path, reach, cols, total_time


def smooth_q_path(q_path, window=5, polyorder=3):
    """
    Apply Savitzky-Golay filter to each joint independently.
    window   : must be odd and > polyorder
    polyorder: polynomial order for the filter
    """
    q_path = np.asarray(q_path)
    q_smooth = np.zeros_like(q_path)
    for j in range(q_path.shape[1]):
        q_smooth[:, j] = savgol_filter(q_path[:, j], window_length=window, polyorder=polyorder)
    return q_smooth
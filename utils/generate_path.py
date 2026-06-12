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

        # ------------------------------------------------------------------ #
        # PHASE 1 — collect all IK candidates for every waypoint             #
        # ------------------------------------------------------------------ #
        all_candidates = []   # [n] -> np.array (M_j, nu)  or empty
        all_col_mask   = []   # [n] -> np.array (M_j,) bool  True = collision

        for j in range(n):
            t_w_p = cartesian_path[j][0]
            theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 = cartesian_path[j][1]

            #! Handle task redundancy
            sols_ok, fk_ok = [], []
            for i in range(ik_params.N_disc):
                R_w_p_rotated = R.from_euler('XYZ',[theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 + i * 2 * np.pi / ik_params.N_disc],degrees=False).as_matrix()

                A_w_p_rotated = np.eye(4)
                A_w_p_rotated[:3, 3] = t_w_p
                A_w_p_rotated[:3, :3] = R_w_p_rotated

                A_b_wl3 = (
                    np.linalg.inv(A_w_b)
                    @ A_w_p_rotated
                    @ np.linalg.inv(A_ee_t)
                    @ np.linalg.inv(A_wl3_ee)
                )
                quat_pose  = rotm_to_quaternion(A_b_wl3[:3, :3])
                target     = np.array([
                    A_b_wl3[0, 3], A_b_wl3[1, 3], A_b_wl3[2, 3],
                    quat_pose[0],  quat_pose[1],   quat_pose[2], quat_pose[3]
                ], dtype=np.float64)

                tgt_tensor = torch.from_numpy(target.astype(np.float32))
                sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N=ik_params.N_samples, fast_solver=fast_ik_solver)
                sols_ok.append(sols_disc)
                fk_ok.append(fk_disc)

            sols_np = torch.cat(sols_ok, dim=0).cpu().numpy()   # (M, nu)

            if sols_np.shape[0] == 0:
                all_candidates.append(np.empty((0, rob_par.nu)))
                all_col_mask.append(np.empty(0, dtype=bool))
                continue

            # Collision mask for every candidate
            col_mask = np.zeros(len(sols_np), dtype=bool)
            for i, q in enumerate(sols_np):
                data.qpos[:rob_par.nu] = q.tolist()
                mujoco.mj_forward(model, data)
                col_mask[i] = get_collisions(model, data, False) > 0

            all_candidates.append(sols_np) #* Joint configurations for all candidates (including those in collision)
            all_col_mask.append(col_mask) #* Boolean mask indicating which candidates are in collision

        # ------------------------------------------------------------------ #
        # PHASE 2 — build collision-free candidate sets                       #
        # ------------------------------------------------------------------ #
        # free_candidates[j] -> np.array (F_j, nu)  only collision-free configs
        # If F_j == 0, the waypoint is unreachable; keep one dummy config
        # so indexing never breaks, and flag it.

        free_candidates = []
        waypoint_reachable = []

        for j in range(n): # n = number of waypoints
            cands = all_candidates[j]
            col_mask = all_col_mask[j]

            if cands.shape[0] == 0:
                free_candidates.append(np.empty((0, rob_par.nu)))
                waypoint_reachable.append(False)
            else:
                free = cands[~col_mask]
                if free.shape[0] == 0:
                    # All candidates collide — keep full set as fallback
                    # (will be flagged as collision in output)
                    free_candidates.append(cands)
                    waypoint_reachable.append(False)
                else:
                    free_candidates.append(free)
                    waypoint_reachable.append(True)

        # ------------------------------------------------------------------ #
        # PHASE 3 — find the most constrained waypoint                        #
        #                                                                     #
        # "Most constrained" = fewest collision-free candidates.              #
        # This is our anchor: we pick its best config first (smallest         #
        # displacement from q_home), then propagate outward in both           #
        # directions choosing the nearest neighbour at each step.             #
        # ------------------------------------------------------------------ #

        # Count free candidates per waypoint
        free_counts = np.array([fc.shape[0] for fc in free_candidates])

        # Most constrained waypoint index
        anchor_idx = int(free_counts.argmin())

        # Among anchor's candidates, pick the one closest to q_home
        anchor_cands = free_candidates[anchor_idx]   # (F, nu)
        q_home = rob_par.home_configuration

        anchor_displacements = np.sum(
            (anchor_cands - q_home[np.newaxis, :]) ** 2, axis=1
        )
        anchor_best_local = int(anchor_displacements.argmin())
        anchor_best_q     = anchor_cands[anchor_best_local]

        #print(f"{fonts.blue}Home configuration: {q_home}, best starting configuration: {anchor_best_q}{fonts.reset}")
        #sys.exit()

        # ------------------------------------------------------------------ #
        # PHASE 4 — greedy propagation in both directions from anchor         #
        # ------------------------------------------------------------------ #
        # chosen[j] will hold the selected np.array (nu,) for waypoint j

        chosen = [None] * n
        chosen[anchor_idx] = anchor_best_q

        def pick_nearest(q_ref, candidates):
            """Return the candidate with smallest wrapped L2 distance from q_ref."""
            diff = candidates - q_ref[np.newaxis, :]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            displacements = np.sum(diff ** 2, axis=1)
            best = int(displacements.argmin())
            return candidates[best]

        # Propagate LEFT  (anchor_idx-1  down to  0)
        for j in range(anchor_idx - 1, -1, -1):
            q_ref = chosen[j + 1]
            chosen[j] = pick_nearest(q_ref, free_candidates[j])

        # Propagate RIGHT (anchor_idx+1  up to  n-1)
        for j in range(anchor_idx + 1, n):
            q_ref = chosen[j - 1]
            chosen[j] = pick_nearest(q_ref, free_candidates[j])

        # ------------------------------------------------------------------ #
        # PHASE 5 — assemble output in original waypoint order                #
        # ------------------------------------------------------------------ #
        for j in range(n):
            q = chosen[j]
            # Determine collision status from original full candidate set
            cands = all_candidates[j]
            col_mask = all_col_mask[j]

            if cands.shape[0] == 0:
                # Completely unreachable
                reach.append(1)
                cols.append(0)
                q_path.append(q)
                continue

            # Find which original candidate matches chosen[j]
            diff = cands - q[np.newaxis, :]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            dists = np.sum(diff ** 2, axis=1)
            match_idx = int(dists.argmin())
            is_col    = bool(col_mask[match_idx])

            reach.append(0 if waypoint_reachable[j] else 1)
            cols.append(1 if is_col else 0)
            q_path.append(q)

            # Keep MuJoCo state consistent
            data.qpos[:rob_par.nu] = q.tolist()
            mujoco.mj_forward(model, data)

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
                t_w_p, target_rot, q_init=q_old
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
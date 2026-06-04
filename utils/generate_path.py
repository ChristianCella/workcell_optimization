import sys, time, os, mujoco, torch
from mujoco_utils import *
from transformations import *   
import fonts

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


def create_path(cartesian_path, model, data, tool_tip_site_id, A_w_b, A_ee_t, A_wl3_ee, save_data):
    start_time = time.time()
    q_path = []
    cols = []
    reach = []
    for j in range(len(cartesian_path)):
        q_old = data.qpos[:rob_params.nu].copy()
        t_w_p = cartesian_path[j][0]
        theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 = cartesian_path[j][1]           

        #! Time-consuming solver
        if ik_solver_to_use == "ikflow":
            sols_ok, fk_ok = [], []
            start_time = time.time()
            #* Retrieve N ik solutions
            for i in range(ik_params.N_disc):
                R_w_p_rotated = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 + i * 2 * np.pi / ik_params.N_disc], degrees=False).as_matrix()
                A_w_p_rotated = np.eye(4)
                A_w_p_rotated[:3, 3] = t_w_p
                A_w_p_rotated[:3, :3] = R_w_p_rotated
                A_b_wl3 = np.linalg.inv(A_w_b) @ A_w_p_rotated @ np.linalg.inv(A_ee_t)@ np.linalg.inv(A_wl3_ee)
                quat_pose = rotm_to_quaternion(A_b_wl3[:3, :3])
                target = np.array([
                    A_b_wl3[0, 3], A_b_wl3[1, 3], A_b_wl3[2, 3],   # position
                    quat_pose[0], quat_pose[1], quat_pose[2], quat_pose[3]  # quaternion
                ], dtype=np.float64)
                tgt_tensor = torch.from_numpy(target.astype(np.float32))
                sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N = ik_params.N_samples, fast_solver=fast_ik_solver) # Find N solutions for this target
                sols_ok.append(sols_disc)
                fk_ok.append(fk_disc)

            end_time = time.time()
            #print(f"{fonts.blue}Trajectory point {j+1}{fonts.reset}")
            #print(f"{fonts.green}IK solutions computed in {end_time - start_time:.2f} seconds, that is {(end_time - start_time)/60:.2f} minutes{fonts.reset}")

            # bring solutions back to host for numpy()
            sols_ok = torch.cat(sols_ok, dim=0)
            fk_ok = torch.cat(fk_ok, dim=0)
            sols_np = sols_ok.cpu().numpy()
            fk_np = fk_ok.cpu().numpy()

            #! Check reachability
            if sols_np.shape[0] == 0:
                #print(f"{fonts.red}No IK solutions found for trajectory point {j+1}!{fonts.reset}")
                reach.append(1) # Mark as failure
            else:
                reach.append(0) # Mark as success

            #* Smallest joint displacement
            start_time = time.time()
            best_cost = 1e12
            best_q = np.zeros(rob_params.nu)
            for i, (q, x) in enumerate(zip(sols_np, fk_np), 1):

                # apply joint solution
                data.qpos[:6] = q.tolist()
                mujoco.mj_forward(model, data)
                n_cols = get_collisions(model, data, False)

                # Smallest joint displacement
                displacement = joint_displacement(q, q_old)
                if (displacement < best_cost) and (n_cols == 0):
                    best_cost = displacement
                    best_q = q

            # Found optimal config
            #print(f"{fonts.yellow}Best solution for trajectory point {j+1} found in {time.time() - start_time:.2f} seconds!{fonts.reset}")
            q_path.append(best_q)
            cols.append(1 if (best_cost == 1e12 and sols_np.shape[0] != 0) else 0) # 1 if no solution found, 0 otherwise
            data.qpos[:6] = best_q.tolist()
            mujoco.mj_forward(model, data)

        #! Damped-least squares method
        elif ik_solver_to_use == "dls":
            target_rot = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0], degrees=False).as_matrix()
            best_q, res = solve_ik_dls(model, data, tool_tip_site_id, t_w_p, target_rot, q_init=q_old)
            data.qpos[:6] = best_q
            mujoco.mj_forward(model, data)
            q_path.append(best_q)
            reach.append(res) 
        else:
            raise ValueError(f"Unknown IK solver type: {ik_solver_to_use}")
        

    # Path found
    q_path = np.unwrap(q_path, axis=0) #! Remove multiplicity
    total_time = time.time() - start_time

    if save_data:
        # Save path to CSV
        q_path_np = np.array(q_path)  
        csv_path = os.path.join(base_dir, "workcell_optimization/results", f"q_path_{robot_to_use}_{ik_solver_to_use}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        np.savetxt(csv_path, q_path_np, delimiter=",",
                   header="q1,q2,q3,q4,q5,q6", comments="")
        print(f"{fonts.green}Path saved to {csv_path}{fonts.reset}")

    return q_path, reach, cols, total_time
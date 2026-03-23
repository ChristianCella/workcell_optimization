#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os
import torch
import csv
from pathlib import Path
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from parameters import UseCaseData, Ur5eRobot, Tools, TestIkFlow
from create_scene import create_reference_frames, merge_robot_and_tool, inject_robot_tool_into_scene, add_instance
params = TestIkFlow()
from create_scene import create_reference_frames, merge_robot_and_tool, inject_robot_tool_into_scene, add_instance

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(base_dir)
import fonts
from transformations import rotm_to_quaternion, quaternion_to_euler, quaternion_to_rpy, get_homogeneous_matrix
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, compute_jacobian
from ikflow_inference import FastIKFlowSolver, solve_ik_fast

# Half-ranges and mid-ranges of the robot
opt_par = UseCaseData()
rob_par = Ur5eRobot()
tool_par = Tools()
m = 0.5 * (rob_par.lb + rob_par.ub)
s = 0.5 * (rob_par.ub - rob_par.lb)


def save_manip_results_to_csv(results, output_csv_path):
    fieldnames = ["test_id", "pose_x", "pose_y", "theta_deg", "manip", "centering"]

    with open(output_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():

    # Path setup
    tool_filename = "screwdriver_marco.xml"
    robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    obstacle_name = "plate.xml"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # CSV output path
    output_csv_path = os.path.join(os.path.dirname(__file__), "manip_results.csv")

    # Create the robot + tool model
    _ = merge_robot_and_tool(
        tool_filename=tool_filename,
        base_dir=base_dir,
        output_robot_tool_filename=robot_and_tool_file_name
    )

    # Add the robot + tool to the scene
    merged_scene_path = inject_robot_tool_into_scene(
        robot_tool_filename=robot_and_tool_file_name,
        output_scene_filename=output_scene_filename,
        base_dir=base_dir
    )

    # Add a piece for screwing
    obstacle_path = os.path.join(base_dir, "ur5e_utils_mujoco/screwing_pieces", obstacle_name)
    add_instance(
        merged_scene_path,
        obstacle_path,
        merged_scene_path,
        mesh_source_dir=os.path.join(base_dir, "ur5e_utils_mujoco/screwing_pieces"),
        mesh_target_dir=os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/assets")
    )

    # Create the reference frames
    temp_xml_name = create_reference_frames(base_dir, "ur5e_utils_mujoco/" + output_scene_filename, 1)
    model_path = os.path.join(base_dir, "ur5e_utils_mujoco", temp_xml_name)

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    component_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reference_target_1")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')
    screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")
    wrist_3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

    # All fixed quantities
    poses_x = [-0.485, -0.5, -0.325, -0.62, -0.473, -0.555, -0.515, -0.515, -0.335, -0.455, -0.29, -0.63, -0.295, -0.292, -0.41, -0.41, -0.28, -0.29, -0.615, -0.285]
    poses_y = [0.252, 0.25, 0.163, 0.34, 0.395, 0.425, 0.365, 0.07, 0.355, 0.23, 0.325, 0.155, -0.01, -0.003, 0.41, 0.133, 0.27, 0.385, 0.235, -0.02]
    thetas = [-30.0, -60.0, 0.0, 45.0, 90.0, -30.0, 0.0, -30.0, 45.0, -30.0, 0.0, -60.0, -90.0, 90.0, 90.0, -90.0, 45.0, 0.0, -30.0, -60.0]

    # Collect CSV rows here
    manip_results = []

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Scan all the poses
        for idx, posex in enumerate(poses_x):

            posey = poses_y[idx]
            theta = thetas[idx]

            # Set robot base (matrix A^w_b)
            t_w_b = np.array([0.0, 0.0, 0.0])
            R_w_b = R.from_euler('XYZ', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
            A_w_b = np.eye(4)
            A_w_b[:3, 3] = t_w_b
            A_w_b[:3, :3] = R_w_b
            set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

            # Set the piece in the environment out of the way (matrix A^w_p)
            _, _, A_w_p = get_homogeneous_matrix(1.5, 1.5, 0.0, 0.0, 0.0, 0.0)
            set_body_pose(model, data, component_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

            theta_for_csv = theta

            if theta == 90.0:
                theta = 0.0
                t_ee_t1 = np.array([0.0, 0.0, 0.0])
                R_ee_t1 = R.from_euler('XYZ', [np.radians(0.0), np.radians(0.0), np.radians(-45.0)], degrees=False).as_matrix()
                A_ee_t1 = np.eye(4)
                A_ee_t1[:3, 3] = t_ee_t1
                A_ee_t1[:3, :3] = R_ee_t1
            else:
                # Set the frame 'screw_top to a new pose wrt flange' and move the screwdriver there
                t_ee_t1 = np.array([-0.06, 0.0, 0.0455])
                R_ee_t1 = R.from_euler('XYZ', [np.radians(0), np.radians(-90), np.radians(90)], degrees=False).as_matrix()
                A_ee_t1 = np.eye(4)
                A_ee_t1[:3, 3] = t_ee_t1
                A_ee_t1[:3, :3] = R_ee_t1

            # Rotate the frame of theta deg around x
            fixed_radius = 0.0455
            t_t1_t2 = np.array([0.0, fixed_radius - (fixed_radius * np.cos(np.radians(theta))), -fixed_radius * np.sin(np.radians(theta))])
            R_t1_t2 = R.from_euler('XYZ', [np.radians(theta), 0, 0], degrees=False).as_matrix()
            A_t1_t2 = np.eye(4)
            A_t1_t2[:3, 3] = t_t1_t2
            A_t1_t2[:3, :3] = R_t1_t2

            # Compute the pose of the screwdriver body
            A_ee_t2 = A_ee_t1 @ A_t1_t2
            set_body_pose(model, data, screwdriver_body_id, A_ee_t2[:3, 3], rotm_to_quaternion(A_ee_t2[:3, :3]))

            # Fixed transformation 'tool top (t1) => tool tip (t)'
            t_t2_t = np.array([0, -0.195, 0.028])
            R_t2_t = R.from_euler('XYZ', [np.radians(90), np.radians(0), np.radians(0)], degrees=False).as_matrix()
            A_t2_t = np.eye(4)
            A_t2_t[:3, 3] = t_t2_t
            A_t2_t[:3, :3] = R_t2_t

            # Update the position of the tool tip
            A_ee_t = A_ee_t1 @ A_t1_t2 @ A_t2_t
            set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

            # End-effector with respect to wrist3
            t_wl3_ee = np.array([0, 0.1, 0])
            R_wl3_e = R.from_euler('XYZ', [np.radians(-90), 0, 0], degrees=False).as_matrix()
            A_wl3_ee = np.eye(4)
            A_wl3_ee[:3, 3] = t_wl3_ee
            A_wl3_ee[:3, :3] = R_wl3_e

            # IKFlow inference
            fast_ik_solver = FastIKFlowSolver()
            counter_start_inference = time.time()

            # Piece in the world
            theta_w_p_x_0 = np.radians(180)
            theta_w_p_y_0 = np.radians(0)
            theta_w_p_z_0 = np.radians(90)
            t_w_p = np.array([posex, posey, 0.03])
            R_w_p = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0], degrees=False).as_matrix()
            A_w_p = np.eye(4)
            A_w_p[:3, 3] = t_w_p
            A_w_p[:3, :3] = R_w_p

            # Loop through the discrete configurations
            sols_ok, fk_ok = [], []
            for i in range(params.N_disc):
                R_w_p_rotated = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 + i * 2 * np.pi / params.N_disc], degrees=False).as_matrix()
                A_w_p_rotated = np.eye(4)
                A_w_p_rotated[:3, 3] = t_w_p
                A_w_p_rotated[:3, :3] = R_w_p_rotated
                A_b_wl3 = np.linalg.inv(A_w_b) @ A_w_p_rotated @ np.linalg.inv(A_ee_t) @ np.linalg.inv(A_wl3_ee)
                quat_pose = rotm_to_quaternion(A_b_wl3[:3, :3])
                target = np.array([
                    A_b_wl3[0, 3], A_b_wl3[1, 3], A_b_wl3[2, 3],
                    quat_pose[0], quat_pose[1], quat_pose[2], quat_pose[3]
                ], dtype=np.float64)
                tgt_tensor = torch.from_numpy(target.astype(np.float32))
                sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N=params.N_samples, fast_solver=fast_ik_solver)
                sols_ok.append(sols_disc)
                fk_ok.append(fk_disc)

            sols_ok = torch.cat(sols_ok, dim=0)
            fk_ok = torch.cat(fk_ok, dim=0)
            sols_np = sols_ok.cpu().numpy()
            fk_np = fk_ok.cpu().numpy()

            # Update marker pose
            quat_frame = rotm_to_quaternion(A_w_p[:3, :3])
            set_body_pose(
                model, data, piece_body_id,
                t_w_p.tolist(),
                [quat_frame[0], quat_frame[1], quat_frame[2], quat_frame[3]]
            )
            mujoco.mj_forward(model, data)

            # Loop over each valid IK solution
            best_cost_fol = 1e12
            best_q = None

            for i, (q, x) in enumerate(zip(sols_np, fk_np), 1):

                # Apply joint solution
                data.qpos[:6] = q.tolist()
                mujoco.mj_forward(model, data)

                n_cols = get_collisions(model, data, False)
                fit_fol_prim = inverse_manipulability(q, model, data, tool_site_id)

                # Follower secondary objective
                diff = q.copy() - m
                fit_fol_sec = float(np.sum(opt_par.centering_weights * (diff / s) ** 2))

                # Total cost for the j-th follower
                cost_fol = opt_par.weights_follower[0] * fit_fol_prim + opt_par.weights_follower[1] * fit_fol_sec

                # Save the configuration with best inverse manipulability
                if (cost_fol < best_cost_fol) and (n_cols == 0):
                    best_cost_fol = cost_fol
                    best_q = q.copy()

            if best_q is None:
                print(f"Test {idx+1}: no collision-free IK solution found.")
                manip = np.nan
            else:
                print(f"The best configuration is: {np.round(best_q, 3)} with cost {best_cost_fol:.3f}")
                #input("Press Enter to apply the best configuration…")

                # Optimization is over => apply the best configuration
                data.qpos[:6] = best_q.tolist()
                mujoco.mj_forward(model, data)
                viewer.sync()

                # Manipulability
                manip = float(inverse_manipulability(best_q, model, data, tool_site_id))
                z = (data.qpos[:rob_par.nu] - m) / s
                s_val = 1.0 - np.abs(z)
                s_mean = np.mean(s_val)
                print(f"Manipulability: {manip:.6f}, Centering: {s_mean:.6f}")

            # Save one row for this iteration
            manip_results.append({
                "test_id": idx + 1,
                "pose_x": posex,
                "pose_y": posey,
                "theta_deg": theta_for_csv,
                "manip": manip,
                "centering": s_mean if best_q is not None else np.nan
            })

    # Write CSV once at the end
    save_manip_results_to_csv(manip_results, output_csv_path)
    print(f"\nSaved manipulability results to: {output_csv_path}")


if __name__ == "__main__":
    main()
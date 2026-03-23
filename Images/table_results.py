#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import csv
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from parameters import UseCaseData, Ur5eRobot, Tools
from create_scene import create_reference_frames, merge_robot_and_tool, inject_robot_tool_into_scene, add_instance

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(base_dir)
import fonts
from transformations import rotm_to_quaternion, get_homogeneous_matrix
from mujoco_utils import set_body_pose, compute_jacobian, inverse_manipulability
from ikflow_inference import FastIKFlowSolver, solve_ik_fast

opt_par = UseCaseData()
rob_par = Ur5eRobot()
tool_par = Tools()

# Half-ranges and mid-ranges of the robot
m = 0.5 * (rob_par.lb + rob_par.ub)
s = 0.5 * (rob_par.ub - rob_par.lb)


def set_joint_configuration(data, model, desired_qpos):
    # Set the position and velocity to 0
    data.qpos[:model.nq] = desired_qpos.copy()
    data.qvel[:model.nq] = 0.0
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.0


def save_results_to_csv(results, output_csv_path):
    fieldnames = [
        "test_id",
        "theta_deg",
        "q1", "q2", "q3", "q4", "q5", "q6",
        "tau_grav_1", "tau_grav_2", "tau_grav_3", "tau_grav_4", "tau_grav_5", "tau_grav_6",
        "tau_ext_1", "tau_ext_2", "tau_ext_3", "tau_ext_4", "tau_ext_5", "tau_ext_6",
        "tau_tot_1", "tau_tot_2", "tau_tot_3", "tau_tot_4", "tau_tot_5", "tau_tot_6",
        "tau_tot_norm",
        "inv_manip",
        "delta_q"
    ]

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

    # Output CSV path
    output_csv_path = os.path.join(os.path.dirname(__file__), "simulation_metrics_results.csv")

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
    verbose = True

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Store all results here
        results = []

        # Target poses for the robot
        target_qpos_list = [
            # Optimal
            [2.7364097, 4.532557, 2.1198652, -0.8659327, -1.5944375, -4.7210255],

            # Participant 1
            [2.416611433029175, -1.181342439060547, 1.8226121107684534, -2.7360946140685023, -1.563422981892721, -1.5698941389666956],
            [2.4844958782196045, -1.2819624704173584, 1.7984798590289515, -1.864180704156393, -2.616671148930685, 0.2440221607685089],
            [2.526808023452759, -1.799110551873678, 2.4054487387286585, -2.1866799793639125, -1.5542267004596155, 2.6722517013549805],

            # Participant 2
            [2.4481801986694336, -0.9696028989604493, 1.8829601446734827, -3.2900835476317347, -1.554371182118551, 1.5778119564056396],
            [2.0026228427886963, -1.0648410481265564, 1.7185271422015589, -1.0928587478450318, 0.03870123624801636, 1.2479023933410645],
            [-0.4898927847491663, -2.101778646508688, -1.4220973253250122, -1.3172285717776795, -5.223900381718771, 0.29893508553504944],

            # Participant 3
            [2.3076064586639404, -1.3282729250243683, 1.951245133076803, -2.2300759754576625, -1.5758350531207483, 1.5766067504882812],
            [-3.5672689119922083, -1.4758638379028817, 1.9072426001178187, -1.4604488101652642, -1.6062710920916956, 1.4203931093215942],
            [1.991189956665039, -1.317534552221634, 2.4801812807666224, -3.4961310825743617, -1.43516713777651, 1.5230153799057007],

            # Participant 4
            [2.3769195079803467, -1.4986998003772278, 1.9541118780719202, -1.5739895306029261, -1.8926995436297815, 0.987227201461792],
            [-4.282005612050192, -1.802681108514303, 2.4401212374316614, -2.253880640069479, -1.5709131399737757, 1.5627403259277344],
            [-0.03496677080263311, -2.223323961297506, -1.0721105337142944, -3.4726692638792933, -1.5749638716327112, 1.5528371334075928],
            [-3.53563100496401, -1.6154786549010218, 2.074448887501852, -0.46930380285296636, -1.8047030607806605, 1.5534722805023193],
            [-3.4957101980792444, -1.2961570781520386, 2.0434592405902308, -0.7268748444369812, -1.7992284933673304, 0.7903189659118652],

            # Participant 5
            [-4.010712210332052, -1.1549570125392457, 1.8205154577838343, -0.6829689306071778, -4.01833159128298, -5.476731363927023],
            [2.5405704975128174, -0.9825790089419861, 1.862021271382467, -4.0208240948119105, -4.5875564257251185, -1.570761505757467],
            [2.3156652450561523, -1.4183880549720307, 2.090379063283102, 0.2600084978291015, -4.1824145952807825, 5.327007293701172],

            # Participant 6
            [-0.4787958304034632, -1.5473455873182793, -2.225069284439087, -0.9409183424762269, 1.602348804473877, 2.7341020107269287],
            [2.5827910900115967, -1.1822010439685364, 1.5426915327655237, -1.4115844529918213, -1.5672181288348597, 1.625714898109436],
            [0.6282005310058594, -1.1744966965964814, -2.2862865924835205, -2.1413399181761683, 0.845022439956665, 1.0461125373840332]
        ]

        # Vector of thetas
        thetas = [-60, -30.0, -60.0, 0.0, 45.0, 90.0, -30.0, 0.0, -30.0, 45.0, -30.0, 0.0, -60.0, -90.0, 90.0, 90.0, -90.0, 45.0, 0.0, -30.0, -60.0]

        # Define bodies, geometries and sites
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        tool_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")
        tool_tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
        tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')
        piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")

        # Always use the site's parent body for COM and for xfrc_applied
        site_parent_body = model.site_bodyid[tool_tip_site_id]
        parent_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, site_parent_body)
        if verbose:
            print(f"{fonts.red}The parent body name is: {parent_body_name}{fonts.reset}")

        # External wrench defined in the world frame, applied at the tool site
        external_force_world = np.array([0.0, 0.0, 0.0])
        external_torque_world = np.array([0.0, 0.0, -2.0])
        site_wrench_world = np.hstack([external_force_world, external_torque_world])

        print(f"The current joints are (deg): {np.round(np.degrees(data.qpos[:6]), 2)}")

        with mujoco.viewer.launch_passive(model, data) as viewer:

            # Scan all the poses
            for idx, desired_qpos in enumerate(target_qpos_list):

                print(f"Test {idx+1}")
                mujoco.mj_resetData(model, data)

                desired_qpos = np.array(desired_qpos, dtype=float)

                # Set the new robot base (matrix A^w_b)
                _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

                # Set the piece in the environment (matrix A^w_p)
                _, _, A_w_p = get_homogeneous_matrix(2.0, 2.0, -0.02, 0.0, 0.0, 0.0)
                set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

                # Set the base of the tool with respect to the flange
                theta = thetas[idx]
                theta_for_csv = theta
                fixed_radius = 0.0455

                if theta == 90.0:
                    _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, -45.0)
                    theta = 0.0
                    print(f"Screwdriver directly in the flange. Using theta = {theta}")
                else:
                    _, _, A_ee_t1 = get_homogeneous_matrix(-0.06, 0.0, 0.0455, 0.0, -90.0, 90.0)

                _, _, A_t1_t2 = get_homogeneous_matrix(
                    0.0,
                    fixed_radius - (fixed_radius * np.cos(np.radians(theta))),
                    -fixed_radius * np.sin(np.radians(theta)),
                    theta, 0.0, 0.0
                )

                A_ee_t2 = A_ee_t1 @ A_t1_t2
                set_body_pose(model, data, tool_base_body_id, A_ee_t2[:3, 3], rotm_to_quaternion(A_ee_t2[:3, :3]))

                # Fixed transformation 'tool top (t1) => tool tip (t)'
                _, _, A_t2_t = get_homogeneous_matrix(0, -0.195, 0.028, 90.0, 0.0, 0.0)
                A_ee_t = A_ee_t1 @ A_t1_t2 @ A_t2_t
                set_body_pose(model, data, tool_tip_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

                # Set the robot in the specified configuration
                set_joint_configuration(data, model, desired_qpos)

                # Simulate for 2 seconds
                seconds_per_config = 2.0
                start_time = time.perf_counter()

                mujoco.mj_forward(model, data)
                viewer.sync()

                # Initialize values so they are available after the while loop
                tau_ext = np.zeros(6)
                gravity_comp = np.zeros(6)
                tau_tot = np.zeros(6)
                tau_hat_abs = 0.0
                inv_manip = 0.0
                delta_q = 0.0

                while time.perf_counter() - start_time < seconds_per_config:
                    mujoco.mj_forward(model, data)

                    # Adjustments required by mujoco
                    p_site = data.site_xpos[tool_tip_site_id]
                    p_com = data.xipos[site_parent_body]
                    r = p_site - p_com

                    # Shift torque to COM: T_com = T_site + r x F
                    F_site = site_wrench_world[:3]
                    T_site = site_wrench_world[3:]
                    T_com = T_site + np.cross(r, F_site)

                    # Apply the reaction wrench at the parent body's COM (world-frame)
                    data.xfrc_applied[site_parent_body, :3] = -F_site
                    data.xfrc_applied[site_parent_body, 3:] = -T_com

                    # Compute torques
                    J6 = compute_jacobian(model, data, tool_tip_site_id)
                    tau_ext = J6.T @ site_wrench_world
                    gravity_comp = data.qfrc_bias[:6].copy()
                    data.ctrl[:6] = gravity_comp + tau_ext

                    # Leader metric
                    tau_tot = gravity_comp + tau_ext
                    tau_hat_abs = float(np.linalg.norm(tau_tot))

                    # Compute manipulability
                    inv_manip = float(inverse_manipulability(desired_qpos.copy(), model, data, tool_tip_site_id))

                    # Joint centering
                    z = (data.qpos[:rob_par.nu] - m) / s
                    s_val = 1.0 - np.abs(z)
                    delta_q = np.mean(s_val)

                    mujoco.mj_step(model, data)
                    viewer.sync()

                    # Clear xfrc_applied so we reapply explicitly each loop
                    data.xfrc_applied[site_parent_body, :] = 0.0

                # Print debug
                if verbose:
                    print(f"{fonts.green}tau_grav: {np.round(gravity_comp, 2)}{fonts.reset}")
                    print(f"{fonts.cyan}tau_ext (J^T w): {np.round(tau_ext, 2)}{fonts.reset}")
                    print(f"{fonts.yellow}tau_tot: {np.round(tau_tot, 2)}{fonts.reset}")
                    print(f"{fonts.blue}||tau_tot||: {np.round(tau_hat_abs, 2)}{fonts.reset}")
                    print(f"{fonts.purple}Inv. manip: {inv_manip}{fonts.reset}")
                    print(f"{fonts.red}Joints centering: {delta_q}{fonts.reset}")

                # Save row for CSV
                row = {
                    "test_id": idx + 1,
                    "theta_deg": theta_for_csv,

                    "q1": desired_qpos[0],
                    "q2": desired_qpos[1],
                    "q3": desired_qpos[2],
                    "q4": desired_qpos[3],
                    "q5": desired_qpos[4],
                    "q6": desired_qpos[5],

                    "tau_grav_1": gravity_comp[0],
                    "tau_grav_2": gravity_comp[1],
                    "tau_grav_3": gravity_comp[2],
                    "tau_grav_4": gravity_comp[3],
                    "tau_grav_5": gravity_comp[4],
                    "tau_grav_6": gravity_comp[5],

                    "tau_ext_1": tau_ext[0],
                    "tau_ext_2": tau_ext[1],
                    "tau_ext_3": tau_ext[2],
                    "tau_ext_4": tau_ext[3],
                    "tau_ext_5": tau_ext[4],
                    "tau_ext_6": tau_ext[5],

                    "tau_tot_1": tau_tot[0],
                    "tau_tot_2": tau_tot[1],
                    "tau_tot_3": tau_tot[2],
                    "tau_tot_4": tau_tot[3],
                    "tau_tot_5": tau_tot[4],
                    "tau_tot_6": tau_tot[5],

                    "tau_tot_norm": tau_hat_abs,
                    "inv_manip": inv_manip,
                    "delta_q": delta_q
                }
                results.append(row)

            # Save CSV after all tests
            save_results_to_csv(results, output_csv_path)
            print(f"\nResults saved to: {output_csv_path}")
            print("\n--- Finished all configurations. ---")
            input("Press Enter to close the viewer and exit...")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
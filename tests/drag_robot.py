#!/usr/bin/env python3
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
import os, sys
import tkinter as tk

#* Utils directory
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
import fonts
from mujoco_utils import scene_manager, get_cartesian_pose


CONTROL_MODE = "joints" # "joints" or "cartesian"
BODY_NAME = "tool_tip_proxy" 
N_JOINTS = 6

# Robot model path (your existing logic)
ur5e_utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ur5e_utils_mujoco'))
model_path = scene_manager("robot", 1, ur5e_utils_dir, "bringup_ur5e.xml", "extension.xml")

''' 
Functions.
'''

def rpy_to_matrix(roll, pitch, yaw):
    """Convert roll, pitch, yaw (XYZ, in radians) to a 3x3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])

    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])

    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])

    # R = Rz * Ry * Rx
    return Rz @ Ry @ Rx

''' 
Viewer threads.
'''
def viewer_thread_joints(model, data, q_target, n_joints, running_flag):
    """Viewer thread: purely kinematic, joints set directly from q_target."""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last = time.time()
        while viewer.is_running() and running_flag["running"]:
            # Apply the target joints (first n_joints) kinematically
            data.qpos[:n_joints] = q_target[:n_joints]
            mujoco.mj_forward(model, data)

            viewer.sync()

            if time.time() - last > 0.5:
                q_deg = np.degrees(data.qpos[:n_joints])
                print(f"{fonts.yellow}q (deg): {np.round(q_deg, 2)}{fonts.reset}")
                last = time.time()

            time.sleep(0.01)

    running_flag["running"] = False


def viewer_thread_cartesian(model, data, body_id, p0, R0, pos_offset, rpy_offset,
                            n_joints, running_flag):
    """
    Viewer thread: Cartesian control with IK on a specific body.
    pos_offset: [dx, dy, dz] in meters (sliders)
    rpy_offset: [droll, dpitch, dyaw] in radians (sliders)
    """
    nv = model.nv
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last = time.time()

        # Buffers for Jacobian
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))

        while viewer.is_running() and running_flag["running"]:
            # Desired pose of the body
            dx, dy, dz = pos_offset[:]          # meters
            dro, dpi, dya = rpy_offset[:]       # radians

            p_des = p0 + np.array([dx, dy, dz])
            R_delta = rpy_to_matrix(dro, dpi, dya)
            R_des = R0 @ R_delta

            # Run a few small IK iterations per frame
            for _ in range(5):
                mujoco.mj_forward(model, data)

                # Current pose
                p_cur = data.xpos[body_id].copy()
                R_flat = data.xmat[body_id].copy()
                R_cur = R_flat.reshape(3, 3)

                # Position error
                e_p = p_des - p_cur

                # Orientation error via rotation matrix (small-angle approx)
                R_err = R_des @ R_cur.T
                # Skew-symmetric part → vector (axis * sin(theta))
                e_r = 0.5 * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])

                e = np.concatenate([e_p, e_r])  # 6x1

                # Jacobian of body
                jacp[:] = 0.0
                jacr[:] = 0.0
                mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
                J6 = np.vstack((jacp, jacr))           # (6, nv)
                J = J6[:, :n_joints]                   # only first n_joints

                # Damped least squares / gradient step
                alpha = 0.5
                dq = alpha * J.T @ e                   # (n_joints,)

                # Limit max step per iteration
                max_step = np.deg2rad(2.0)
                dq = np.clip(dq, -max_step, max_step)

                data.qpos[:n_joints] += dq

            mujoco.mj_forward(model, data)
            viewer.sync()

            if time.time() - last > 0.5:
                q_deg = np.degrees(data.qpos[:n_joints])
                # Get the forward kinematics at a specified frame
                pos, quat = get_cartesian_pose(body_id, data, "quaternion")
                print(f"{fonts.green}FK: pos={np.round(pos, 3)}, quat={np.round(quat, 3)}{fonts.reset}")
                print(f"{fonts.yellow}q (deg): {np.round(q_deg, 2)}{fonts.reset}")
                last = time.time()

            time.sleep(0.01)

    running_flag["running"] = False


''' 
GUI functions.
'''

def create_gui_joints(model, data, q_target, n_joints, running_flag):
    """Tk GUI with 1 slider per joint (degrees)."""
    root = tk.Tk()
    root.title("UR5e Joint Sliders")

    tk.Label(root, text="Move sliders to control joints (degrees)").pack(pady=5)

    # Try to use joint limits; fallback to [-180, 180]
    for j in range(n_joints):
        if j < model.njnt:
            low, high = model.jnt_range[j]
            if low == 0.0 and high == 0.0:
                low_deg, high_deg = -180.0, 180.0
            else:
                low_deg, high_deg = np.degrees([low, high])
        else:
            low_deg, high_deg = -180.0, 180.0

        frame = tk.Frame(root)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame, text=f"Joint {j}").pack(anchor="w")

        init_deg = float(np.degrees(q_target[j]))

        def make_callback(idx):
            def on_change(val):
                angle_deg = float(val)
                q_target[idx] = np.deg2rad(angle_deg)
            return on_change

        slider = tk.Scale(
            frame,
            from_=high_deg,
            to=low_deg,
            orient="horizontal",
            resolution=0.5,
            length=300,
            command=make_callback(j),
        )
        slider.set(init_deg)
        slider.pack(fill="x")

    def on_close():
        running_flag["running"] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def create_gui_cartesian(p_offset, rpy_offset, running_flag, body_name):
    """
    Tk GUI with 6 sliders:
      - 3 for position offsets [dx, dy, dz] in meters
      - 3 for orientation offsets [droll, dpitch, dyaw] in degrees
    """
    root = tk.Tk()
    root.title(f"Cartesian Control: {body_name}")

    tk.Label(root, text=f"Cartesian offsets for body '{body_name}'").pack(pady=5)

    # Position sliders: +/- 0.5 m
    labels_pos = ["dx (m)", "dy (m)", "dz (m)"]
    for i, lab in enumerate(labels_pos):
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame, text=lab).pack(anchor="w")

        def make_pos_cb(idx):
            def on_change(val):
                p_offset[idx] = float(val)
            return on_change

        slider = tk.Scale(
            frame,
            from_=0.5,
            to=-0.5,
            orient="horizontal",
            resolution=0.001,
            length=500,
            command=make_pos_cb(i),
        )
        slider.set(0.0)
        slider.pack(fill="x")

    # Orientation sliders: +/- 90 deg
    labels_rot = ["droll (deg)", "dpitch (deg)", "dyaw (deg)"]
    for i, lab in enumerate(labels_rot):
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame, text=lab).pack(anchor="w")

        def make_rot_cb(idx):
            def on_change(val):
                rpy_offset[idx] = np.deg2rad(float(val))
            return on_change

        slider = tk.Scale(
            frame,
            from_=90.0,
            to=-90.0,
            orient="horizontal",
            resolution=1.0,
            length=300,
            command=make_rot_cb(i),
        )
        slider.set(0.0)
        slider.pack(fill="x")

    def on_close():
        running_flag["running"] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

''' 
Main function.
'''

def main():
    # Load model + data
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    # Kinematic usage
    model.opt.gravity[:] = 0.0
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_init = np.radians([-90, -90, -90, -90, 90, 0])
    data.qpos[:6] = q_init
    mujoco.mj_forward(model, data)

    running_flag = {"running": True}

    mode = CONTROL_MODE.lower()
    if mode == "joints":
        # Joint mode
        n_joints = min(N_JOINTS, model.nq)
        q_target = np.copy(data.qpos[:n_joints])

        print("\nKinematic JOINT control with sliders")
        print("  • MuJoCo viewer shows the robot")
        print("  • Tk GUI provides 1 slider per joint (degrees)\n")

        vt = threading.Thread(
            target=viewer_thread_joints,
            args=(model, data, q_target, n_joints, running_flag),
            daemon=True,
        )
        vt.start()

        create_gui_joints(model, data, q_target, n_joints, running_flag)
        running_flag["running"] = False
        vt.join(timeout=1.0)

    elif mode == "cartesian":
        # Cartesian mode
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
        if body_id < 0:
            raise RuntimeError(f"Body '{BODY_NAME}' not found in model")

        mujoco.mj_forward(model, data)
        p0 = data.xpos[body_id].copy()
        R0_flat = data.xmat[body_id].copy()
        R0 = R0_flat.reshape(3, 3)

        pos_offset = np.zeros(3)      # dx, dy, dz
        rpy_offset = np.zeros(3)      # droll, dpitch, dyaw (rad)

        n_joints = min(N_JOINTS, model.nq)

        print("\nKinematic CARTESIAN control with sliders")
        print(f"  • Controlling body: {BODY_NAME}")
        print("  • 3 sliders: dx, dy, dz (m)")
        print("  • 3 sliders: droll, dpitch, dyaw (deg)")
        print("  • Simple Jacobian IK moves joints so body tracks pose\n")

        vt = threading.Thread(
            target=viewer_thread_cartesian,
            args=(model, data, body_id, p0, R0, pos_offset, rpy_offset,
                  n_joints, running_flag),
            daemon=True,
        )
        vt.start()

        create_gui_cartesian(pos_offset, rpy_offset, running_flag, BODY_NAME)
        running_flag["running"] = False
        vt.join(timeout=1.0)

    else:
        raise ValueError("CONTROL_MODE must be 'joints' or 'cartesian'")

    print("Exiting.")


if __name__ == "__main__":
    main()

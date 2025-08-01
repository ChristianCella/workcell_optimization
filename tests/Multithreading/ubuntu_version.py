import mujoco
import mujoco.viewer
import numpy as np
import multiprocessing as mp
import os
import sys
import time
import torch

from ikflow_inference import FastIKFlowSolver, solve_ik_fast
from transformations import rotm_to_quaternion

# === CONFIGURATION ===
USE_VIEWER = True        # Show Mujoco viewers
N = 8                    # Number of poses
DOF = 6                  # UR5e DOF
VIEW_TIME = 4            # Viewer duration in seconds
XML_PATH = os.path.abspath("path/to/ur5e.xml")  # <-- CHANGE THIS

# === HELPER FUNCTIONS ===
def generate_random_cartesian_pose():
    """Generate a random 4x4 pose matrix."""
    position = np.random.uniform(low=[0.3, -0.5, 0.1], high=[0.7, 0.5, 0.6])
    rotation = np.eye(3)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose

def pose_to_tensor(pose):
    """Convert 4x4 pose to [x, y, z, qx, qy, qz, qw] tensor."""
    position = pose[:3, 3]
    rotation = pose[:3, :3]
    quat = rotm_to_quaternion(rotation)  # assumed to return [x, y, z, w]
    return torch.tensor([*position, *quat], dtype=torch.float32)

# === SHARED SOLVER (inherited on Linux via fork) ===
solver = None

def initialize_solver():
    global solver
    if solver is None:
        solver = FastIKFlowSolver()
        print("[INFO] FastIKFlowSolver initialized in main process.")

# === WORKER FUNCTION ===
def ik_and_evaluate(pose, idx, result_queue, use_viewer):
    global solver

    try:
        tgt_tensor = pose_to_tensor(pose)
        ik_solutions, _ = solve_ik_fast(tgt_tensor, N=1, fast_solver=solver)

        if ik_solutions is None or ik_solutions.numel() == 0:
            print(f"[{idx}] No IK solution found.")
            result_queue.put((idx, 0.0))
            return

        joint_angles = ik_solutions[0].cpu().numpy()

        # MuJoCo simulation
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
        data.qpos[:DOF] = joint_angles
        mujoco.mj_forward(model, data)

        # Compute manipulability
        end_effector = "wrist_3_link"
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, model.body(end_effector).id)
        J = np.vstack([jacp, jacr])
        JJ_T = J @ J.T
        try:
            manipulability = float(np.sqrt(np.linalg.det(JJ_T)))
        except np.linalg.LinAlgError:
            manipulability = 0.0

        result_queue.put((idx, manipulability))

        # Optional viewer
        if use_viewer:
            with mujoco.viewer.launch_passive(model, data, title=f"Robot {idx}") as viewer:
                start = time.time()
                while viewer.is_running() and time.time() - start < VIEW_TIME:
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(0.01)

    except Exception as e:
        print(f"[{idx}] Error: {e}")
        result_queue.put((idx, 0.0))

# === MAIN ===
if __name__ == "__main__":
    mp.set_start_method("fork")  # Ubuntu default, but just to be sure

    print("[INFO] Initializing solver...")
    initialize_solver()

    print("[INFO] Generating Cartesian target poses...")
    target_poses = [generate_random_cartesian_pose() for _ in range(N)]

    print("[INFO] Launching multiprocessing workers...")
    result_queue = mp.Queue()
    processes = []

    for i, pose in enumerate(target_poses):
        p = mp.Process(target=ik_and_evaluate, args=(pose, i, result_queue, USE_VIEWER))
        p.start()
        processes.append(p)

    results = [None] * N
    for _ in range(N):
        idx, value = result_queue.get()
        results[idx] = value

    for p in processes:
        p.join()

    result_vector = np.array(results)
    print("\n[RESULT] Batch manipulability vector:")
    print(result_vector)

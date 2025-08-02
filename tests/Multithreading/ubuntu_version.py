import os
import sys
import time
import torch
import numpy as np
import multiprocessing as mp
import mujoco
import mujoco.viewer

'''
For how fork and spawn work, we cannot use GPU for batch evalautions.
'''

# === PATH SETUP ===
solver_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils'))
sys.path.append(solver_dir)
from ikflow_inference import FastIKFlowSolver, solve_ik_fast
from transformations import rotm_to_quaternion

# === CONFIGURATION ===
USE_VIEWER = False       # Set to True to enable GUI
N = 8                    # Number of poses to evaluate
DOF = 6                  # UR5e DOF
VIEW_TIME = 4            # Viewer time per robot
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
XML_PATH = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/ur5e.xml")

# === HELPERS ===
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
    quat = rotm_to_quaternion(rotation)
    return torch.tensor([*position, *quat], dtype=torch.float32)

# === WORKER FUNCTION ===
def ik_and_evaluate_cpu(pose, idx, result_queue, use_viewer):
    try:
        # Create local solver (force CPU)
        solver = FastIKFlowSolver()

        # Solve IK
        tgt_tensor = pose_to_tensor(pose).to('cpu')
        ik_solutions, _ = solve_ik_fast(tgt_tensor, N=3000, fast_solver=solver)

        if ik_solutions is None or ik_solutions.numel() == 0:
            print(f"[{idx}] No IK solution found.")
            result_queue.put((idx, 0.0))
            return

        joint_angles = ik_solutions[0].cpu().numpy()

        # Load model and simulate
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
            with mujoco.viewer.launch_passive(model, data) as viewer:
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
    mp.set_start_method("spawn", force=True)  # Safe for torch + multiprocessing

    print("[INFO] Generating random poses...")
    target_poses = [generate_random_cartesian_pose() for _ in range(N)]

    print("[INFO] Spawning workers...")
    result_queue = mp.Queue()
    processes = []

    start_time = time.time()
    print("[INFO] Starting IK inference...")
    for i, pose in enumerate(target_poses):
        p = mp.Process(target=ik_and_evaluate_cpu, args=(pose, i, result_queue, USE_VIEWER))
        p.start()
        processes.append(p)
    

    results = [None] * N
    for _ in range(N):
        idx, value = result_queue.get()
        results[idx] = value

    for p in processes:
        p.join()

    end_time = time.time()
    print(f"The batch evaluation lasted {end_time - start_time:.2f} seconds")

    result_vector = np.array(results)
    print("\n[RESULT] Batch manipulability vector:")
    print(result_vector)

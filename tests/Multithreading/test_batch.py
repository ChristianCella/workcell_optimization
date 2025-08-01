import mujoco
import mujoco.viewer
import numpy as np
import multiprocessing as mp
import os, sys
import time

# === CONFIGURATION ===
USE_VIEWER = True  # Set to False to disable GUI viewer
N = 20              # Number of parallel evaluations
DOF = 6            # Number of joints
VIEW_TIME = 4     # Viewer duration in seconds (if enabled)

# === PATH SETUP ===
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_dir)
XML_PATH = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/ur5e.xml")

# === GENERATE RANDOM CONFIGURATIONS ===
joint_configs = [np.random.uniform(-1, 1, DOF) for _ in range(N)]

def launch_viewer_and_compute(joint_angles, idx, result_queue, use_viewer):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    data.qpos[:len(joint_angles)] = joint_angles
    mujoco.mj_forward(model, data)

    # Compute Jacobian and manipulability
    end_effector = "wrist_3_link"  # Change as needed
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, model.body(end_effector).id)
    J = np.vstack([jacp, jacr])
    JJ_T = J @ J.T
    try:
        manipulability = float(np.sqrt(np.linalg.det(JJ_T)))
    except np.linalg.LinAlgError:
        manipulability = 0.0

    # Send result to main process
    result_queue.put((idx, manipulability))

    # Launch viewer if enabled
    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < VIEW_TIME:
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.01)

if __name__ == "__main__":
    result_queue = mp.Queue()
    processes = []

    for i, config in enumerate(joint_configs):
        p = mp.Process(target=launch_viewer_and_compute, args=(config, i, result_queue, USE_VIEWER))
        p.start()
        processes.append(p)

    # Collect all results
    results = [None] * N
    for _ in range(N):
        idx, value = result_queue.get()
        results[idx] = value

    for p in processes:
        p.join()

    # Convert to numpy vector
    result_vector = np.array(results)
    print("Batch manipulability vector:", result_vector)

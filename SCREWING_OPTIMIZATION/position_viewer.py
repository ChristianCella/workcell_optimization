# position_viewer.py
import os, time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# stato globale
_model = None
_data = None
_viewer = None
_ref_body_ids = []
_base_id = None
_tool_id = None

def euler_to_quat(r, p, y, deg=False):
    q = R.from_euler('xyz', [r, p, y], degrees=deg).as_quat()
    return [q[3], q[0], q[1], q[2]]  # w, x, y, z

def setup_target_frames(model, data, ref_body_ids, target_poses):
    for i, (pos, quat) in enumerate(target_poses):
        model.body_pos[ref_body_ids[i]] = pos
        model.body_quat[ref_body_ids[i]] = [quat[3], quat[0], quat[1], quat[2]]
    mujoco.mj_forward(model, data)

def init_viewer(x0, y0, z0):
    global _model, _data, _viewer, _ref_body_ids, _base_id, _tool_id

    # --- carico il modello ---
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    xml = os.path.join(root, "universal_robots_ur5e", "scene.xml")
    _model = mujoco.MjModel.from_xml_path(xml)
    _data  = mujoco.MjData(_model)

    # --- trovo gli ID dei corpi ---
    # (stesso codice di prima, adattato a riempire _ref_body_ids, _base_id, _tool_id)
    for i in range(3):
        bid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, f"reference_target_{i+1}")
        if bid == -1:
            bid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, "reference_target")
        _ref_body_ids.append(bid)
    _base_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, "base")
    _tool_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")

    # --- posizioni iniziali ---
    q_start = np.radians([-8.38, -68.05, -138, -64, 90, -7.85])
    _model.body_pos[_base_id]  = [0.0, 0.0, 0.0]
    _model.body_quat[_base_id] = euler_to_quat(0,0,0,deg=True)
    _model.body_pos[_tool_id]  = [x0, y0, z0]
    _model.body_quat[_tool_id] = euler_to_quat(0,0,0,deg=True)
    _data.qpos[:6] = q_start

    # --- target frames statici (esempio) ---
    target_poses = [
        (np.array([0.2, -0.2, 0.2]), R.from_euler('xyz', [180, 0, 0], degrees=True).as_quat()),
        (np.array([0.3,  0.1, 0.7]), R.from_euler('xyz', [  0, 0, 0], degrees=True).as_quat()),
        (np.array([0.3,  0.3, 0.3]), R.from_euler('xyz', [135, 0,90], degrees=True).as_quat()),
    ]
    setup_target_frames(_model, _data, _ref_body_ids, target_poses)

    mujoco.mj_forward(_model, _data)

    # --- apro la finestra una volta sola, in passive mode ---
    _viewer = mujoco.viewer.launch_passive(_model, _data)
    time.sleep(0.2)  # lascia un po’ di tempo al GLFW di inizializzarsi

def update_viewer(x, y, z):
    """Chiama questa funzione dopo ogni nuova valutazione x_next."""
    global _model, _data, _viewer, _tool_id
    # aggiorno solo il tool_frame (o la base, come preferisci)
    _model.body_pos[_base_id] = [x, y, z]
    mujoco.mj_forward(_model, _data)
    _viewer.sync()   # ridisegna la scena

def close_viewer():
    global _viewer
    if _viewer is not None:
        _viewer.close()
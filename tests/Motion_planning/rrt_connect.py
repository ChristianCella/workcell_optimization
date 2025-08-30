import sys, os
import numpy as np
import mujoco 
import mujoco.viewer
import math
import time
from dataclasses import dataclass

# Append the path to 'scene_manager'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scene_manager')))
from create_scene import create_scene

# Append the path to 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from transformations import rotm_to_quaternion, get_homogeneous_matrix
from mujoco_utils import set_body_pose

#! general utilities
def clamp_to_limits(q, limits):
    """Clamp each joint angle/position to [low, high]."""
    q_clamped = np.clip(q, limits[:, 0], limits[:, 1])
    return q_clamped

def in_limits(q, limits):
    return np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1])

def angdiff(a, b):
    """Angle difference for revolute joints with limits typically present."""
    d = (b - a + np.pi) % (2*np.pi) - np.pi
    return d

def weighted_distance(q1, q2, weights=None, revolute_mask=None):
    """Euclidean distance with optional joint weights and angle-wrap on revolutes."""
    if revolute_mask is None:
        revolute_mask = np.ones_like(q1, dtype=bool)
    d = np.where(revolute_mask, angdiff(q1, q2), (q2 - q1))
    if weights is None:
        return np.linalg.norm(d)
    return np.linalg.norm(weights * d)

def interpolate(q1, q2, step, weights=None, revolute_mask=None):
    """Take one step from q1 toward q2 in joint space using the weighted metric."""
    if revolute_mask is None:
        revolute_mask = np.ones_like(q1, dtype=bool)
    d = np.where(revolute_mask, angdiff(q1, q2), (q2 - q1))
    L = np.linalg.norm(d if weights is None else weights * d)
    if L < 1e-12:
        return q1.copy(), 0.0
    alpha = min(1.0, step / L)
    return (q1 + alpha * d), (L - step)

#! MuJoCo utilities
class MuJoCoCollisionChecker:
    def __init__(self, model, base_qpos=None, joint_ids=None):
        """
        model: mujoco.MjModel
        base_qpos: reference qpos to start from (others fixed); if None uses zeros.
        joint_ids: the joint indices (0..njnt-1) we plan over (e.g., first 6).
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.base_qpos = np.zeros(model.nq) if base_qpos is None else np.array(base_qpos, dtype=float).copy()

        if joint_ids is None:
            self.joint_ids = np.arange(model.njnt, dtype=int)
        else:
            self.joint_ids = np.array(joint_ids, dtype=int)

        # Map each planned joint to its qpos address (hinge/slide => 1 dof each)
        self.qpos_addr = self.model.jnt_qposadr[self.joint_ids]
        # NEW: map planned joints to DOF columns (for Jacobians)
        self.dof_mask = np.zeros(self.model.nv, dtype=bool)
        self.dof_mask[self.model.jnt_dofadr[self.joint_ids]] = True

    def set_qpos_for_planned_joints(self, q):
        """Write planned joints into data.qpos on top of base_qpos."""
        self.data.qpos[:] = self.base_qpos  # reset other joints
        self.data.qpos[self.qpos_addr] = q

    def in_collision(self, q):
        """Return True if any contact exists for configuration q."""
        self.set_qpos_for_planned_joints(q)
        mujoco.mj_forward(self.model, self.data)
        return self.data.ncon > 0
    
    def edge_collision_free(self, q1, q2, per_joint_step=0.02, weights=None, revolute_mask=None):
        """
        Check the straight edge q1->q2 by interpolating.
        per_joint_step: max step per joint (radians for hinges, meters for slides).
        """
        if revolute_mask is None:
            revolute_mask = np.ones_like(q1, dtype=bool)

        delta = np.where(revolute_mask, angdiff(q1, q2), (q2 - q1))
        max_delta = np.max(np.abs(delta))
        n_steps = max(1, int(math.ceil(max_delta / per_joint_step)))

        for i in range(1, n_steps + 1):
            q = q1 + (i / n_steps) * delta
            if self.in_collision(q):
                return False
        return True
    
#! RRT_connect utilities
@dataclass
class Node:
    q: np.ndarray
    parent: int  # index into the node list (-1 for root)

class Tree:
    def __init__(self, q_root):
        self.nodes = [Node(q=q_root.copy(), parent=-1)]

    def add(self, q, parent_idx):
        self.nodes.append(Node(q=q.copy(), parent=parent_idx))
        return len(self.nodes) - 1

    def nearest(self, q_query, weights=None, revolute_mask=None):
        """Linear NN search (simple and fine for small/medium problems)."""
        dmin = float('inf')
        idx = 0
        for i, node in enumerate(self.nodes):
            d = weighted_distance(node.q, q_query, weights, revolute_mask)
            if d < dmin:
                dmin = d
                idx = i
        return idx

    def path_to_root(self, idx):
        P = []
        while idx != -1:
            P.append(self.nodes[idx].q.copy())
            idx = self.nodes[idx].parent
        P.reverse()
        return P

#! RRT connect planner
class RRTConnectPlanner:
    def __init__(self,
                 collision_checker: MuJoCoCollisionChecker,
                 joint_limits: np.ndarray,            # shape (n,2)
                 step_size: float = 0.1,              # step in metric units (rad)
                 per_joint_check_step: float = 0.02,  # rad per joint for edge checking
                 goal_tolerance: float = 0.02,        # rad RMS-ish threshold
                 goal_bias: float = 0.1,
                 max_iters: int = 20000,
                 rng: np.random.Generator | None = None,
                 weights: np.ndarray | None = None,
                 revolute_mask: np.ndarray | None = None,
                 min_progress: float = 1e-6,
                 max_connect_steps: float = 200
                 ):
        self.cc = collision_checker
        self.limits = joint_limits
        self.step = step_size
        self.per_joint_step = per_joint_check_step
        self.goal_tol = goal_tolerance
        self.goal_bias = goal_bias
        self.max_iters = max_iters
        self.rng = np.random.default_rng() if rng is None else rng
        self.weights = weights
        self.revolute_mask = np.ones(joint_limits.shape[0], dtype=bool) if revolute_mask is None else revolute_mask
        self.min_progress = min_progress
        self.max_connect_steps = max_connect_steps

    def sample(self, q_goal):
        if self.rng.random() < self.goal_bias:
            return q_goal.copy()
        low, high = self.limits[:, 0], self.limits[:, 1]
        return self.rng.uniform(low, high)

    def steer(self, q_from, q_to):
        q_new, _ = interpolate(q_from, q_to, self.step, self.weights, self.revolute_mask)
        q_new = clamp_to_limits(q_new, self.limits)
        return q_new

    def extend(self, tree: Tree, q_target):
        """Attempt one step toward q_target. Return ('Trapped'|'Advanced'|'Reached', idx_new or idx_near)."""
        idx_near = tree.nearest(q_target, self.weights, self.revolute_mask)
        q_near = tree.nodes[idx_near].q
        q_new = self.steer(q_near, q_target)

        if weighted_distance(q_new, q_near, self.weights, self.revolute_mask) < self.min_progress:
            return 'Trapped', idx_near

        if not in_limits(q_new, self.limits):
            return 'Trapped', idx_near
        if not self.cc.edge_collision_free(q_near, q_new, self.per_joint_step, self.weights, self.revolute_mask):
            return 'Trapped', idx_near

        idx_new = tree.add(q_new, idx_near)
        if weighted_distance(q_new, q_target, self.weights, self.revolute_mask) <= self.goal_tol:
            return 'Reached', idx_new
        
        return 'Advanced', idx_new

    def connect(self, tree: Tree, q_target):
        """Greedily extend until trapped or reached target neighborhood."""
        steps = 0
        while True:
            status, idx = self.extend(tree, q_target)
            steps += 1
            if status != 'Advanced':
                return status, idx
            if steps >= self.max_connect_steps:
                return 'Trapped', idx

    def plan(self, q_start, q_goal, time_budget_s: float | None = None):
        """Run RRT-Connect and return (path or None, stats dict)."""
        if not in_limits(q_start, self.limits) or not in_limits(q_goal, self.limits):
            raise ValueError("Start or goal outside joint limits.")

        if self.cc.in_collision(q_start):
            raise RuntimeError("Start configuration is in collision.")
        if self.cc.in_collision(q_goal):
            raise RuntimeError("Goal configuration is in collision.")

        Ta = Tree(q_start)
        Tb = Tree(q_goal)

        start_time = time.time()
        iters = 0

        for iters in range(1, self.max_iters + 1):
            if time_budget_s is not None and (time.time() - start_time) > time_budget_s:
                break

            q_rand = self.sample(q_goal)

            status_a, idx_a = self.extend(Ta, q_rand)
            if status_a != 'Trapped':
                qa_new = Ta.nodes[idx_a].q
                status_b, idx_b = self.connect(Tb, qa_new)

                if status_b == 'Reached':
                    # Build path: Ta root -> qa_new + reverse of Tb root -> goal
                    path_a = Ta.path_to_root(idx_a)       # start ... qa_new
                    path_b = Tb.path_to_root(idx_b)       # goal-side ... qa_new
                    path_b_rev = list(reversed(path_b))   # qa_new ... goal
                    if len(path_b_rev) > 0:
                        path_b_rev = path_b_rev[1:]        # remove qa_new
                    path = path_a + path_b_rev

                    stats = dict(
                        iters=iters,
                        time_s=time.time() - start_time,
                        nodes_start=len(Ta.nodes),
                        nodes_goal=len(Tb.nodes)
                    )
                    return path, stats

            # Swap trees
            Ta, Tb = Tb, Ta

        return None, dict(iters=iters, time_s=time.time() - start_time, nodes_start=len(Ta.nodes), nodes_goal=len(Tb.nodes))

#! Densify / smooth / resample helpers
def densify_path(path, target_points, weights=None, revolute_mask=None):
    """
    Interpolates between path waypoints to reach exactly `target_points` points.
    The output includes the first and last points.
    """
    path = [np.array(p, dtype=float) for p in path]
    if target_points <= len(path):
        return path

    if revolute_mask is None:
        revolute_mask = np.ones_like(path[0], dtype=bool)
    if weights is None:
        weights = np.ones_like(path[0], dtype=float)

    def seg_len(a, b):
        d = np.where(revolute_mask, angdiff(a, b), (b - a))
        return np.linalg.norm(weights * d)

    seg_lengths = [seg_len(path[i], path[i+1]) for i in range(len(path)-1)]
    cum_length = [0.0]
    for L in seg_lengths:
        cum_length.append(cum_length[-1] + L)
    total_length = cum_length[-1]

    new_points = [path[0]]
    for k in range(1, target_points-1):
        target_s = (k / (target_points-1)) * total_length
        for i in range(len(seg_lengths)):
            if target_s <= cum_length[i+1]:
                seg_ratio = (target_s - cum_length[i]) / seg_lengths[i]
                q_interp = path[i] + seg_ratio * np.where(
                    revolute_mask, angdiff(path[i], path[i+1]), (path[i+1] - path[i])
                )
                new_points.append(q_interp)
                break
    new_points.append(path[-1])
    return new_points

def shortcut_smooth(path, cc: MuJoCoCollisionChecker, attempts=100,
                    per_joint_check_step=0.02, weights=None, revolute_mask=None):
    """
    Randomly try to replace subpaths with straight edges if collision-free.
    Keeps start and goal the same.
    """
    if path is None or len(path) < 3:
        return path
    path = [p.copy() for p in path]
    n = len(path)
    for _ in range(attempts):
        if n < 3:
            break
        i = np.random.randint(0, n - 2)
        j = np.random.randint(i + 2, n)
        if cc.edge_collision_free(path[i], path[j], per_joint_check_step, weights, revolute_mask):
            path = path[:i+1] + path[j:]
            n = len(path)
    return path

def prune_near_duplicates(path, min_step=1e-3, weights=None, revolute_mask=None):
    """Remove consecutive points whose joint-space distance < min_step."""
    if len(path) <= 1:
        return path
    out = [path[0]]
    for q in path[1:]:
        if weighted_distance(out[-1], q, weights, revolute_mask) >= min_step:
            out.append(q)
    return out

def resample_path_by_count(path, target_points, weights=None, revolute_mask=None):
    """Evenly resample the path to exactly target_points along joint-space arc length."""
    path = [np.array(p, float) for p in path]
    if target_points <= 2 or len(path) <= 2:
        return [path[0], path[-1]] if target_points == 2 else path

    if revolute_mask is None:
        revolute_mask = np.ones_like(path[0], dtype=bool)
    if weights is None:
        weights = np.ones_like(path[0], dtype=float)

    def seg_vec(a, b):
        return np.where(revolute_mask, angdiff(a, b), (b - a))

    def seg_len(a, b):
        d = seg_vec(a, b)
        return np.linalg.norm(weights * d)

    seg_lengths = [seg_len(path[i], path[i+1]) for i in range(len(path)-1)]
    cum = [0.0]
    for L in seg_lengths:
        cum.append(cum[-1] + L)
    total = cum[-1]
    if total <= 1e-12:
        return [path[0]] * target_points  # degenerate

    out = [path[0]]
    for k in range(1, target_points-1):
        s = (k / (target_points-1)) * total
        i = np.searchsorted(cum, s, side="right") - 1
        i = min(max(i, 0), len(seg_lengths)-1)
        t = (s - cum[i]) / (seg_lengths[i] if seg_lengths[i] > 0 else 1.0)
        q = path[i] + t * seg_vec(path[i], path[i+1])
        out.append(q)
    out.append(path[-1])
    return out

# ================= NEW: manipulability-aware optimization ===================

def manipulability_yoshikawa(cc: MuJoCoCollisionChecker, q: np.ndarray, site_id: int) -> float:
    """
    Yoshikawa manipulability sqrt(det(J J^T)) at a site, using only planned DOFs.
    """
    model, data = cc.model, cc.data
    cc.set_qpos_for_planned_joints(q)
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack((jacp, jacr))[:, cc.dof_mask]   # (6, n_plan)

    JJt = J @ J.T
    # Robust determinant (could be near-singular)
    try:
        val = np.sqrt(max(np.linalg.det(JJt), 0.0))
    except np.linalg.LinAlgError:
        val = 0.0
    return float(max(val, 1e-12))

def path_length_cost(path, weights=None, revolute_mask=None):
    return sum(
        weighted_distance(path[k], path[k+1], weights, revolute_mask)
        for k in range(len(path)-1)
    )

def smoothness_cost(path):
    if len(path) < 3:
        return 0.0
    c = 0.0
    for k in range(1, len(path)-1):
        c += np.linalg.norm(path[k+1] - 2*path[k] + path[k-1])**2
    return c

def singularity_cost(cc: MuJoCoCollisionChecker, path, site_id: int, psi_eps=1e-4):
    """
    Sum of -log(manipulability + eps) along the path.
    """
    c = 0.0
    for k in range(len(path)):
        w = manipulability_yoshikawa(cc, path[k], site_id)
        c += -math.log(w + psi_eps)
    return c

def total_path_cost(cc: MuJoCoCollisionChecker, path, weights, revolute_mask,
                    site_id: int, alpha=1.0, beta=1e-2, gamma=1.0, psi_eps=1e-4):
    return (
        alpha * path_length_cost(path, weights, revolute_mask) +
        beta  * smoothness_cost(path) +
        gamma * singularity_cost(cc, path, site_id, psi_eps)
    )

def optimize_path_against_singularity(path, cc: MuJoCoCollisionChecker,
                                      joint_limits: np.ndarray,
                                      site_id: int,
                                      weights=None, revolute_mask=None,
                                      alpha=1.0, beta=1e-2, gamma=1.0,
                                      step_init=0.03, step_shrink=0.5,
                                      passes=4, iters_per_waypoint=20,
                                      per_joint_check_step=0.05):
    """
    Local, collision-aware coordinate descent on intermediate waypoints only.
    Accepts a tentative move iff both adjacent edges remain collision-free
    and the total cost decreases.
    """
    path = [np.array(p, float).copy() for p in path]
    if len(path) < 3:
        return path

    best_cost = total_path_cost(cc, path, weights, revolute_mask, site_id,
                                alpha=alpha, beta=beta, gamma=gamma)

    for _ in range(passes):
        improved_any = False
        for k in range(1, len(path)-1):  # keep endpoints
            qk = path[k].copy()
            step = step_init

            for _ in range(iters_per_waypoint):
                improved_here = False
                for j in range(len(qk)):
                    for sgn in (-1.0, +1.0):
                        q_try = path[k].copy()
                        q_try[j] += sgn * step
                        q_try = clamp_to_limits(q_try, joint_limits)

                        # Check feasibility of adjacent edges
                        if not cc.edge_collision_free(path[k-1], q_try, per_joint_check_step, weights, revolute_mask):
                            continue
                        if not cc.edge_collision_free(q_try, path[k+1], per_joint_check_step, weights, revolute_mask):
                            continue

                        # Tentative cost
                        old_qk = path[k]
                        path[k] = q_try
                        cost_try = total_path_cost(cc, path, weights, revolute_mask, site_id,
                                                   alpha=alpha, beta=beta, gamma=gamma)
                        if cost_try + 1e-10 < best_cost:
                            best_cost = cost_try
                            qk = q_try
                            improved_here = True
                            improved_any = True
                        else:
                            path[k] = old_qk  # revert

                if not improved_here:
                    step *= step_shrink
                    if step < 1e-4:
                        break

            path[k] = qk

        if not improved_any:
            break

    return path

def workspace_length_simple(cc: MuJoCoCollisionChecker, path, site_id: int):
    """
    Computes straight-line distance between EE positions for each waypoint in path.
    No interpolation; just FK at waypoints.
    """
    model, data = cc.model, cc.data

    def site_pos(q):
        cc.set_qpos_for_planned_joints(q)
        mujoco.mj_forward(model, data)
        return np.array(data.site_xpos[site_id], dtype=float)

    total = 0.0
    p_prev = site_pos(path[0])
    for q in path[1:]:
        p_curr = site_pos(q)
        total += np.linalg.norm(p_curr - p_prev)
        p_prev = p_curr

    return total


# ========================== /NEW optimization section ========================

#! Test code
if __name__ == "__main__":

    # Variables
    display_gui = True
    n_pieces = 4

    # Path setup 
    tool_filename = "screwdriver.xml"
    robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    piece_name = "table_grip.xml" 
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.append(base_dir)

    # Create the xml scene
    XML_PATH = create_scene(tool_name=tool_filename, robot_and_tool_file_name=robot_and_tool_file_name,
                              output_scene_filename=output_scene_filename, piece_name=piece_name, base_dir=base_dir)

    # Load model and create data
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Decide which joints to plan (here: first N_joints hinge/slide joints)
    n_joints = 6
    assert n_joints <= model.njnt, "The selected # of joints exceeds number of joints in model."
    plan_joint_ids = np.arange(n_joints, dtype=int)

    # Joint limits for planned joints
    jnt_range = model.jnt_range[plan_joint_ids].copy()  # shape (N_joints, 2)

    # Handle unlimited joints (optional defaults)
    for i in range(n_joints):
        if model.jnt_limited[plan_joint_ids[i]] == 0:
            if model.jnt_type[plan_joint_ids[i]] == mujoco.mjtJoint.mjJNT_HINGE:
                jnt_range[i] = np.array([-np.pi, np.pi])
            elif model.jnt_type[plan_joint_ids[i]] == mujoco.mjtJoint.mjJNT_SLIDE:
                jnt_range[i] = np.array([-0.5, 0.5])

    # Boolean mask telling the planner which joints are revolute
    revolute_mask = np.array([model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in plan_joint_ids], dtype=bool)

    # Uniform weights (tune if you want workspace isotropy)
    weights = np.ones(n_joints, dtype=float)

    # Base pose for other (non-planned) joints
    base_qpos = data.qpos.copy()   # keep current defaults for everything else

    # Get body/site IDs
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_grip")
    ref_body_ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("hole_") and name.endswith("_frame_body"):
            ref_body_ids.append(i)
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")

    # Collision checker
    cc = MuJoCoCollisionChecker(model, base_qpos=base_qpos, joint_ids=plan_joint_ids)

    # Joint configurations
    q0 = np.array([1.93, -2.98, 1.41, -1.49, -1.56, 2.44])
    q1 = np.array([-5.835175, 4.402239, -1.5875205, -1.7714313, -4.6415086, -6.1713014])
    q2 = np.array([2.1754599, 3.9184961, -2.262606, -4.605288, 5.1739826, 4.9312534])
    q3 = np.array([-3.39, 4.97, 1.46, 5.25, -1.1965, -0.91])
    q4 = np.array([1.36, -2.36, -0.94, -2.58, -5.29, 3.82])
    q_vec = [q0, q1, q2, q3, q4]

    # set the robot pose
    _, _, A_w_b = get_homogeneous_matrix(0.2664672921246696, 0.068153650497219, 0, 0, 0, 0)
    set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

    # Set the piece in the environment (matrix A^w_p)
    _, _, A_w_p = get_homogeneous_matrix(-0.15, -0.15, 0, 0, 0, 0)
    set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

    # Set the frame 'screw_top to a new pose wrt flange' and move the screwdriver there
    _, _, A_ee_t1 = get_homogeneous_matrix(0, 0.15, 0, 30, 0, 0)
    set_body_pose(model, data, screwdriver_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

    # Fixed transformation 'tool top (t1) => tool tip (t)' (NOTE: the rotation around z is not important)
    _, _, A_t1_t = get_homogeneous_matrix(0, 0, 0.32, 0, 0, 0)

    # Update the position of the tool tip (Just for visualization purposes)
    A_ee_t = A_ee_t1 @ A_t1_t
    set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

    # End-effector with respect to wrist3 (NOTE: this is always fixed)
    _, _, A_wl3_ee = get_homogeneous_matrix(0, 0.1, 0, -90, 0, 0)

    # Planner
    planner = RRTConnectPlanner(
        collision_checker=cc,
        joint_limits=jnt_range,
        step_size=0.15,                 # radians (approx joint metric)
        per_joint_check_step=0.2,       # coarse for planning (OK if geometry is chunky)
        goal_tolerance=0.03,            # ~1.7 deg
        goal_bias=0.15,
        max_iters=500,
        weights=weights,
        revolute_mask=revolute_mask
    )

    # Tests a for loop
    for i in range(n_pieces + 1): # 0, 1, 2, 3, 4
        k = i + 1
        if i == n_pieces:
            k = 0
        # Define start and goal
        q_start = clamp_to_limits(q_vec[i], jnt_range)
        q_goal  = clamp_to_limits(q_vec[k], jnt_range)

        # Set the start joint config
        data.qpos[:6] = q_start.tolist()
        mujoco.mj_forward(model, data)

        path, stats = planner.plan(q_start, q_goal, time_budget_s=5.0)

        # Enforce the path to be start -> goal
        if np.linalg.norm(path[0] - q_start) > np.linalg.norm(path[-1] - q_start):
            path.reverse()

        # Your existing post-processing
        path_pruned = prune_near_duplicates(path, min_step=1e-3,
                                            weights=weights, revolute_mask=revolute_mask)
        path_uniform = resample_path_by_count(path_pruned, target_points=60,
                                              weights=weights, revolute_mask=revolute_mask)        

        # Compute the path length
        L_ee_simple = workspace_length_simple(cc, path_uniform, site_id=tool_site_id)
        print(f"Path {i} length (simple): {L_ee_simple:.4f} m")

    if display_gui:
        # Launch the MuJoCo viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            input("Press enter to start ...")
            for i, q in enumerate(path_uniform):
                data.qpos[:6] = q.tolist()
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.2)
            input("Press enter to continue ...")
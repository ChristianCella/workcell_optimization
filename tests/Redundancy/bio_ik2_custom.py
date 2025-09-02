#!/usr/bin/env python3
import os
# On Windows, use GLFW for GPU rendering
os.environ['MUJOCO_GL'] = 'glfw'

from pyexpat import model
import time
import os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import functools

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils'))
sys.path.append(base_dir)
import fonts
from transformations import rotm_to_quaternion, get_world_wrench, get_homogeneous_matrix, euler_to_quaternion
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, compute_jacobian

manager_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scene_manager'))
sys.path.append(manager_dir)
from create_scene import create_scene


# BioIK2 core solver and goal classes
class Goal:
    ''' 
    Base class for optimization goals in BioIK2.
    '''
    def __init__(self, weight=1.0): self.weight = weight
    def error(self, q, model, data, tool_site_id): raise NotImplementedError

class PoseGoal(Goal):
    '''
    Goal to reach a specific pose with the end-effector.
    '''
    def __init__(self, target_pos, target_z_dir, weight=1.0):
        super().__init__(weight)
        self.target_pos = np.array(target_pos)
        self.target_z   = np.array(target_z_dir)
    def error(self, q, model, data, tool_site_id):
        data.qpos[:model.nv] = q; mujoco.mj_forward(model, data)
        pos   = data.site_xpos[tool_site_id]
        e_pos = np.linalg.norm(pos - self.target_pos)
        z_dir = data.site_xmat[tool_site_id].reshape(3,3)[:,2]
        cos   = np.clip(np.dot(z_dir, self.target_z), -1.0, 1.0)
        ang   = np.arccos(cos)
        return np.sqrt(e_pos**2 + ang**2)

class JointLimitGoal(Goal):
    '''
    Goal to keep joints within specified limits.
    '''
    def __init__(self, joint_lims, weight=1.0):
        super().__init__(weight)
        lims        = np.array(joint_lims)
        self.mid    = 0.5*(lims[:,0] + lims[:,1])
        self.range  = lims[:,1] - lims[:,0]
    def error(self, q, model, data, tool_site_id):
        d = 2*np.abs(q - self.mid) - 0.5*self.range
        return np.linalg.norm(d)

class ElbowGoal(Goal):
    '''
    Goal to keep the elbow joint at a preferred position.
    '''
    def __init__(self, joint_lims, alpha=0.4, weight=1.0):
        super().__init__(weight)
        low, high = joint_lims[3]
        self.pref = low + alpha*(high - low)
    def error(self, q, model, data, tool_site_id):
        return abs(q[3] - self.pref)

class ManipulabilityGoal(Goal):
    '''
    Goal to maximize the manipulability of the end-effector.
    '''
    def error(self, q, model, data, tool_site_id):
        data.qpos[:model.nv] = q; mujoco.mj_forward(model, data)
        Jp = np.zeros((3,model.nv)); Jr = np.zeros((3,model.nv))
        mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
        J = np.vstack([Jp, Jr])[:,:6]
        JJt = J @ J.T
        det = np.linalg.det(JJt)
        return 1e12 if det <= 1e-12 else 1.0/np.sqrt(det)

class AntiAlignGoal(Goal):
    ''' 
    Goal to keep the end-effector's z-axis aligned with a target direction, and not anti-aligned with it. 
    '''
    def __init__(self, target_z, weight=1.0):
        super().__init__(weight)
        self.target_z = np.array(target_z)
    def error(self, q, model, data, tool_site_id):
        data.qpos[:model.nv] = q; mujoco.mj_forward(model, data)
        dot = np.dot(data.site_xmat[tool_site_id].reshape(3,3)[:,2], self.target_z)
        return max(0.0, -dot)

class CollisionGoal(Goal):
    """
    Penalize any penetrating contact in the scene (regardless of which geoms collide).
    Options:
      - exclude_same_body: ignore contacts where both geoms belong to the same body
      - allow_geom_pairs: iterable of (g1, g2) to ignore (unordered)
    """
    def __init__(self, wt=1e5, weight=1.0, exclude_same_body=True, allow_geom_pairs=None):
        super().__init__(weight)
        self.wt = float(wt)
        self.exclude_same_body = bool(exclude_same_body)
        # store as frozenset of 2-tuples with sorted ids for easy membership checks & pickling
        self.allow_geom_pairs = frozenset(
            tuple(sorted(map(int, pair))) for pair in (allow_geom_pairs or [])
        )

    def error(self, q, model, data, tool_site_id):
        # keep signature consistent with other goals
        data.qpos[:model.nv] = q
        mujoco.mj_forward(model, data)

        penalty = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0.0:
                continue  # not penetrating

            g1, g2 = int(c.geom1), int(c.geom2)

            # Skip explicitly allowed pairs
            if tuple(sorted((g1, g2))) in self.allow_geom_pairs:
                continue

            # Optionally skip contacts within the same body (common to co-located geoms)
            if self.exclude_same_body:
                b1 = int(model.geom_bodyid[g1])
                b2 = int(model.geom_bodyid[g2])
                if b1 == b2:
                    continue

            depth = -float(c.dist)                 # penetration depth (meters)
            penalty += depth * 1000.0 + 1.0        # depth-scaled + constant cost per contact

        return self.wt * penalty



# Global function for parallel cost evaluation
def evaluate_cost_parallel(args):
    """
    Global function for parallel cost evaluation
    """
    q, xml_path, tool_site_id, goals_data = args
    
    # Create fresh model and data for this process
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reconstruct goals from serialized data
    goals = []
    for goal_type, goal_params in goals_data:
        if goal_type == 'PoseGoal':
            goals.append(PoseGoal(goal_params['target_pos'], goal_params['target_z'], goal_params['weight']))
        elif goal_type == 'JointLimitGoal':
            goals.append(JointLimitGoal(goal_params['joint_lims'], goal_params['weight']))
        elif goal_type == 'ElbowGoal':
            goals.append(ElbowGoal(goal_params['joint_lims'], goal_params['alpha'], goal_params['weight']))
        elif goal_type == 'ManipulabilityGoal':
            goals.append(ManipulabilityGoal(goal_params['weight']))
        elif goal_type == 'AntiAlignGoal':
            goals.append(AntiAlignGoal(goal_params['target_z'], goal_params['weight']))
        elif goal_type == 'CollisionGoal':
            goals.append(
                CollisionGoal(
                    wt=goal_params['wt'],
                    weight=goal_params['weight'],
                    exclude_same_body=goal_params.get('exclude_same_body', True),
                    allow_geom_pairs=goal_params.get('allow_geom_pairs', []),
                )
            )

    
    # Compute cost
    total_cost = 0.0
    for goal in goals:
        error = goal.error(q, model, data, tool_site_id)
        total_cost += (goal.weight * error) ** 2
    
    return np.sqrt(total_cost)

# Batch evaluation function for threading
def evaluate_costs_batch(qs, xml_path, tool_site_id, goals_data):
    """
    Evaluate multiple costs in a single thread with shared model/data
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reconstruct goals
    goals = []
    for goal_type, goal_params in goals_data:
        if goal_type == 'PoseGoal':
            goals.append(PoseGoal(goal_params['target_pos'], goal_params['target_z'], goal_params['weight']))
        elif goal_type == 'JointLimitGoal':
            goals.append(JointLimitGoal(goal_params['joint_lims'], goal_params['weight']))
        elif goal_type == 'ElbowGoal':
            goals.append(ElbowGoal(goal_params['joint_lims'], goal_params['alpha'], goal_params['weight']))
        elif goal_type == 'ManipulabilityGoal':
            goals.append(ManipulabilityGoal(goal_params['weight']))
        elif goal_type == 'AntiAlignGoal':
            goals.append(AntiAlignGoal(goal_params['target_z'], goal_params['weight']))
        elif goal_type == 'CollisionGoal':
            goals.append(
                CollisionGoal(
                    wt=goal_params['wt'],
                    weight=goal_params['weight'],
                    exclude_same_body=goal_params.get('exclude_same_body', True),
                    allow_geom_pairs=goal_params.get('allow_geom_pairs', []),
                )
            )

    
    costs = []
    for q in qs:
        total_cost = 0.0
        for goal in goals:
            error = goal.error(q, model, data, tool_site_id)
            total_cost += (goal.weight * error) ** 2
        costs.append(np.sqrt(total_cost))
    
    return costs

class BioIK2Solver:
    def __init__(self, model, data, tool_site_id, joint_lims, xml_path,
                 population_size=200, n_elites=5, time_limit=0.1, 
                 use_multiprocessing=True, n_workers=None):
        self.model, self.data, self.tool = model, data, tool_site_id
        self.xml_path = xml_path
        lims = np.array(joint_lims)
        self.lb, self.ub = lims[:,0], lims[:,1]
        self.dim, self.pop_size, self.n_elites = len(self.lb), population_size, n_elites
        self.time_limit, self.goals, self.eval_count = time_limit, [], 0
        
        # Parallelization settings
        self.use_multiprocessing = use_multiprocessing
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        print(f"Using {'multiprocessing' if use_multiprocessing else 'threading'} with {self.n_workers} workers")

    def add_goal(self, goal: Goal):
        self.goals.append(goal)

    def _serialize_goals(self):
        """Serialize goals for parallel processing"""
        goals_data = []
        for goal in self.goals:
            if isinstance(goal, PoseGoal):
                goals_data.append(('PoseGoal', {
                    'target_pos': goal.target_pos,
                    'target_z': goal.target_z,
                    'weight': goal.weight
                }))
            elif isinstance(goal, JointLimitGoal):
                # Reconstruct joint_lims from mid and range
                joint_lims = np.column_stack([
                    goal.mid - 0.5 * goal.range,
                    goal.mid + 0.5 * goal.range
                ])
                goals_data.append(('JointLimitGoal', {
                    'joint_lims': joint_lims,
                    'weight': goal.weight
                }))
            elif isinstance(goal, ElbowGoal):
                # Reconstruct joint_lims - we need to store this info
                # For now, assume standard UR5e limits for joint 4 (elbow)
                joint_lims = [[-2*np.pi, 2*np.pi]] * 6  # Simplified
                goals_data.append(('ElbowGoal', {
                    'joint_lims': joint_lims,
                    'alpha': 0.4,  # Hardcoded from original
                    'weight': goal.weight
                }))
            elif isinstance(goal, ManipulabilityGoal):
                goals_data.append(('ManipulabilityGoal', {
                    'weight': goal.weight
                }))
            elif isinstance(goal, AntiAlignGoal):
                goals_data.append(('AntiAlignGoal', {
                    'target_z': goal.target_z,
                    'weight': goal.weight
                }))
            elif isinstance(goal, CollisionGoal):
                goals_data.append(('CollisionGoal', {
                    'wt': goal.wt,
                    'weight': goal.weight,
                    'exclude_same_body': goal.exclude_same_body,
                    'allow_geom_pairs': [tuple(pair) for pair in goal.allow_geom_pairs],
                }))
        return goals_data

    def cost(self, q):
        """Single cost evaluation (for local optimization)"""
        self.eval_count += 1
        return np.sqrt(sum((g.weight * g.error(q, self.model, self.data, self.tool))**2 
                           for g in self.goals))

    def evaluate_population_parallel(self, population):
        """Evaluate entire population in parallel"""
        goals_data = self._serialize_goals()
        
        if self.use_multiprocessing:
            # Use process-based parallelism
            args = [(q, self.xml_path, self.tool, goals_data) for q in population]
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                costs = list(executor.map(evaluate_cost_parallel, args))
        else:
            # Use thread-based parallelism with batching
            batch_size = max(1, len(population) // self.n_workers)
            batches = [population[i:i+batch_size] for i in range(0, len(population), batch_size)]
            
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(evaluate_costs_batch, batch, self.xml_path, self.tool, goals_data) 
                          for batch in batches]
                batch_costs = [future.result() for future in futures]
                costs = [cost for batch in batch_costs for cost in batch]
        
        self.eval_count += len(population)
        return costs

    def solve(self, tol=1e-2):
        pop = [self.lb + np.random.rand(self.dim)*(self.ub-self.lb) for _ in range(self.pop_size)]
        costs = self.evaluate_population_parallel(pop)

        start = time.perf_counter()
        deadline = start + float(self.time_limit)

        best = min(costs)
        iteration = 0

        while True:
            iteration += 1

            # Hard stop check
            now = time.perf_counter()
            if now >= deadline:
                elapsed = now - start
                print(f"[stop] time limit reached ({elapsed:.3f}s ≥ {self.time_limit}s)")
                break

            if best <= tol:
                print(f"[stop] converged (best={best:.2e} ≤ tol={tol:.2e}) after {iteration-1} iterations")
                break

            # Sort population by fitness
            idx_sort = np.argsort(costs)
            pop = [pop[i] for i in idx_sort]
            costs = [costs[i] for i in idx_sort]
            best = costs[0]

            # --- Elite polishing with time guard ---
            for i in range(self.n_elites):
                if time.perf_counter() >= deadline:
                    break  # no time left to polish further elites
                # keep polish bounded so one elite can't blow the budget
                res = minimize(lambda x: self.cost(x), pop[i],
                            method='L-BFGS-B',
                            bounds=list(zip(self.lb, self.ub)),
                            options={'ftol': 1e-6, 'maxiter': 50})  # small cap
                pop[i] = res.x
                costs[i] = self.cost(res.x)
                best = min(best, costs[i])

            # --- Breed new population ---
            new_pop = pop[:self.n_elites]
            while len(new_pop) < self.pop_size:
                a, b = np.random.choice(self.pop_size, 2, replace=False)
                p1, p2 = pop[a], pop[b]
                alpha = np.random.rand(self.dim)
                child = alpha*p1 + (1-alpha)*p2
                mask = np.random.rand(self.dim) < (1.0/self.dim)
                child[mask] += (np.random.rand(mask.sum())*2-1)*(self.ub-self.lb)[mask]
                new_pop.append(np.clip(child, self.lb, self.ub))

            # --- Evaluate tail in small batches with time guard ---
            tail = new_pop[self.n_elites:]
            new_costs = []
            batch_size = max(1, len(tail) // (self.n_workers or 1) // 2)  # smaller batches
            for i in range(0, len(tail), batch_size):
                if time.perf_counter() >= deadline:
                    break  # stop scheduling new work
                batch = tail[i:i+batch_size]
                new_costs.extend(self.evaluate_population_parallel(batch))

            # If nothing new was evaluated (out of time), stop
            if not new_costs and time.perf_counter() >= deadline:
                elapsed = time.perf_counter() - start
                print(f"[stop] time limit reached ({elapsed:.3f}s ≥ {self.time_limit}s)")
                break

            costs = costs[:self.n_elites] + new_costs
            pop = new_pop
            best = min(best, min(costs))

        print(f"Completed solve(): iterations={iteration}, best_cost={best:.3e}, cost_evals={self.eval_count}")
        return pop[int(np.argmin(costs))]


if __name__ == '__main__':

    # Path setup 
    tool_filename = "screwdriver.xml"
    robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    piece_name = "table_grip.xml" 
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

    # Create the scene
    model_path = create_scene(tool_name=tool_filename, robot_and_tool_file_name=robot_and_tool_file_name,
                              output_scene_filename=output_scene_filename, piece_name=piece_name, base_dir=base_dir)

    # Load the newly created model
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

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
    jlimits = model.jnt_range[:6]

    robot_geoms = [i for i in range(model.ngeom)
                   if 'collision' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')

    #q_start = np.radians([-8.38,-68.05,-138,-64,90,-7.85])
    q_start = np.zeros(6)  # Start from zero configuration
    show_pose_duration = 3.0

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Set the new robot base (matrix A^w_b)
        _, _, A_w_b = get_homogeneous_matrix(0, 0, 0.1, 0, 0, 0)
        set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

        # Set the piece in the environment (matrix A^w_p)
        _, _, A_w_p = get_homogeneous_matrix(0.2, 0.2, 0, 0, 0, 0)
        set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

        # Set the frame 'screw_top to a new pose wrt flange' and move the screwdriver there
        _, _, A_ee_t1 = get_homogeneous_matrix(0, 0, 0.03, np.degrees(0), 0, 0)
        set_body_pose(model, data, screwdriver_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

        # Fixed transformation 'tool top (t1) => tool tip (t)' (NOTE: the rotation around z is not important)
        _, _, A_t1_t = get_homogeneous_matrix(0, 0, 0.32, 0, 0, 0)

        # Update the position of the tool tip (Just for visualization purposes)
        A_ee_t = A_ee_t1 @ A_t1_t
        set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

        # End-effector with respect to wrist3 (NOTE: this is always fixed)
        _, _, A_wl3_ee = get_homogeneous_matrix(0, 0.1, 0, -90, 0, 0)

        # Set the initial configuration
        data.qpos[:6] = q_start       
        mujoco.mj_forward(model, data)
        viewer.sync()

        input("Press Enter to start optimization...")

        for j in range(len(ref_body_ids)):

            #Get the pose of the target 
            posit = data.xpos[ref_body_ids[j]]
            rotm = data.xmat[ref_body_ids[j]].reshape(3, 3)
            quat = R.from_matrix(rotm).as_quat()          
            z_dir = R.from_quat(quat).as_matrix()[:,2]

            # Try multiprocessing first, fall back to threading if issues
            try:
                solver = BioIK2Solver(model, data, tool_site_id, jlimits, model_path,
                                      population_size=200, n_elites=40, time_limit=4.0,
                                      use_multiprocessing=True, n_workers=10)
            except:
                print("Multiprocessing failed, falling back to threading...")
                solver = BioIK2Solver(model, data, tool_site_id, jlimits, model_path,
                                      population_size=3000, n_elites=400, time_limit=2.0,
                                      use_multiprocessing=False, n_workers=10)

            solver.add_goal(PoseGoal(posit, z_dir, weight=1e10))
            solver.add_goal(JointLimitGoal(jlimits, weight=0))
            #solver.add_goal(ElbowGoal(jlimits, alpha=0.4, weight=0))
            solver.add_goal(ManipulabilityGoal(weight=1e3))
            solver.add_goal(AntiAlignGoal(z_dir, weight=1e4))
            solver.add_goal(CollisionGoal(wt=1e4, weight=1.0))

            t0 = time.time()
            q_opt = solver.solve(tol=1e-5)
            print(f"Time: {time.time()-t0:.2f}s")
            print(f"The inverse manipulation metric is {inverse_manipulability(q_opt, model, data, tool_site_id):.2f}")

            data.qpos[:6] = q_opt
            mujoco.mj_forward(model, data)
            viewer.sync()

            # Used in case of more poses
            time.sleep(show_pose_duration)
            viewer.sync()

        input("Optimization sequence completed. Press Enter to exit...")
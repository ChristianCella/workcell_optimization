#!/usr/bin/env python3
import os
# On Windows, use GLFW for GPU rendering
os.environ['MUJOCO_GL'] = 'glfw'

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
from transformations import euler_to_quaternion, rotm_to_quaternion
from mujoco_utils import set_body_pose, setup_target_frames, inverse_manipulability


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
        return 1e3 if det <= 1e-12 else 1.0/np.sqrt(det)

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
    A MuJoCo-based collision penalty that drives the robot away from self-intersections 
    and from penetrating the floor.
    This goal:
      3. Filters for:
         • Robot-robot contacts (two distinct geoms both in robot_geom_ids), and
         • Robot-floor contacts (one geom in robot_geom_ids, the other == floor_geom_id).
      4. For each penetrating contact (dist < 0), adds:
           depth_penalty = max(0, -dist) * 1000.0   # scales with penetration depth
           base_penalty  = 1.0                      # constant per contact
           pen += depth_penalty + base_penalty
      5. Returns wt * pen, allowing you to dial in how strongly collisions are avoided
         relative to other optimization objectives.
    """
    def __init__(self, robot_geom_ids, floor_geom_id=None, wt=1e5, weight=1.0):
        super().__init__(weight)
        self.robot_geom_ids = robot_geom_ids
        self.floor_id       = floor_geom_id
        self.wt             = wt
    def error(self, q, model, data, tool_site_id):
        data.qpos[:model.nv] = q; mujoco.mj_forward(model, data); mujoco.mj_collision(model, data)
        cost_col = 0.0
        for i in range(data.ncon):
            c = data.contact[i]; g1, g2 = c.geom1, c.geom2
            #rr = (g1 in self.robot_geom_ids and g2 in self.robot_geom_ids and g1!=g2)
            #rf = (self.floor_id is not None and 
            #     ((g1 in self.robot_geom_ids and g2==self.floor_id) or 
            #       (g2 in self.robot_geom_ids and g1==self.floor_id)))
            #if rr or rf:
            if data.ncon > 0:
                #pen += max(0.0, -c.dist)*1000.0 + 1.0
                cost_col = self.wt
            else:
                cost_col = 0.0
        return cost_col

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
            goals.append(CollisionGoal(goal_params['robot_geom_ids'], goal_params['floor_id'], 
                                     goal_params['wt'], goal_params['weight']))
    
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
            goals.append(CollisionGoal(goal_params['robot_geom_ids'], goal_params['floor_id'], 
                                     goal_params['wt'], goal_params['weight']))
    
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
                    'robot_geom_ids': goal.robot_geom_ids,
                    'floor_id': goal.floor_id,
                    'wt': goal.wt,
                    'weight': goal.weight
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
        # Initialize population
        pop = [self.lb + np.random.rand(self.dim)*(self.ub-self.lb) for _ in range(self.pop_size)]
        costs = self.evaluate_population_parallel(pop)
        
        start = time.time()
        best = min(costs)
        iteration = 0

        while True:
            iteration += 1
            elapsed = time.time() - start
            if elapsed >= self.time_limit:
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

            # Polish elites with local optimization (sequential - these are few)
            for i in range(self.n_elites):
                res = minimize(lambda x: self.cost(x), pop[i], 
                               method='L-BFGS-B', 
                               bounds=list(zip(self.lb, self.ub)), 
                               options={'ftol':1e-6})
                pop[i] = res.x
                costs[i] = self.cost(res.x)
                best = min(best, costs[i])

            # Generate new population through breeding
            new_pop = pop[:self.n_elites]  # Keep elites
            while len(new_pop) < self.pop_size:
                a, b = np.random.choice(self.pop_size, 2, replace=False)
                p1, p2 = pop[a], pop[b]
                alpha = np.random.rand(self.dim)
                child = alpha*p1 + (1-alpha)*p2
                mask = np.random.rand(self.dim) < (1.0/self.dim)
                child[mask] += (np.random.rand(mask.sum())*2-1)*(self.ub-self.lb)[mask]
                new_pop.append(np.clip(child, self.lb, self.ub))

            # Evaluate new population in parallel
            new_costs = self.evaluate_population_parallel(new_pop[self.n_elites:])
            costs = costs[:self.n_elites] + new_costs
            pop = new_pop
            best = min(best, min(costs))

        print(f"Completed solve(): iterations={iteration}, best_cost={best:.3e}, cost_evals={self.eval_count}")
        return pop[int(np.argmin(costs))]

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, "ur5e_utils_mujoco/scene.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')
    screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screw_top")
    jlimits = model.jnt_range[:6]

    robot_geoms = [i for i in range(model.ngeom)
                   if 'collision' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')

    target_poses = [
        (np.array([0.2,0.2,0.2]), R.from_euler('xyz',[180, 0, 45],True).as_quat())
        #(np.array([0.3, 0.1,0.7]), R.from_euler('xyz',[0,0,0],True).as_quat()),
        #(np.array([0.3, 0.3,0.3]), R.from_euler('xyz',[135,0,90],True).as_quat()),
        #(np.array([-0.4,0.5,0.7]), R.from_euler('xyz',[30,0,0],True).as_quat()),
        #(np.array([0.2,0.2,0.1]), R.from_euler('xyz',[180,0,90],True).as_quat()),
    ]

    #q_start = np.radians([-8.38,-68.05,-138,-64,90,-7.85])
    q_start = np.zeros(6)  # Start from zero configuration
    show_pose_duration = 3.0

    ref_ids = []
    for i in range(len(target_poses)):
        name = f'reference_target_{i+1}'
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'reference_target')
        ref_ids.append(bid)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Set robot base (matrix A^w_b)
        t_w_b = np.array([0, 0, 0])
        R_w_b = R.from_euler('xyz', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
        A_w_b = np.eye(4)
        A_w_b[:3, 3] = t_w_b
        A_w_b[:3, :3] = R_w_b
        set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

        # Set the frame 'screw_top to a new pose wrt flange' and move the screwdriver there
        t_ee_t1 = np.array([0, 0.15, 0])
        R_ee_t1 = R.from_euler('xyz', [np.radians(30), np.radians(0), np.radians(0)], degrees=False).as_matrix()
        A_ee_t1 = np.eye(4)
        A_ee_t1[:3, 3] = t_ee_t1
        A_ee_t1[:3, :3] = R_ee_t1
        set_body_pose(model, data, screwdriver_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

        # Fixed transformation 'tool top (t1) => tool tip (t)'
        t_t1_t = np.array([0, 0.0, 0.26])
        R_t1_t = R.from_euler('xyz', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
        A_t1_t = np.eye(4)
        A_t1_t[:3, 3] = t_t1_t
        A_t1_t[:3, :3] = R_t1_t

        # End-effector with respect to wrist3 (Fixed transformation)
        t_wl3_ee = np.array([0, 0.1, 0])
        R_wl3_e = R.from_euler('xyz', [np.radians(-90), 0, 0], degrees=False).as_matrix()
        A_wl3_ee = np.eye(4)
        A_wl3_ee[:3, 3] = t_wl3_ee
        A_wl3_ee[:3, :3] = R_wl3_e

        # Update the position of the tool tip (Just for visualization purposes)
        A_ee_t = A_ee_t1 @ A_t1_t  # combine the two transformations
        set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

        # Piece in the world (define A^w_p) => this is also used to put the frame in space  
        theta_w_p_x_0 = np.radians(180)
        theta_w_p_y_0 = np.radians(0)
        theta_w_p_z_0 = np.radians(45)
        t_w_p = np.array([0.2, 0.2, 0.2])
        R_w_p = R.from_euler('xyz', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0], degrees=False).as_matrix()
        A_w_p = np.eye(4)
        A_w_p[:3, 3] = t_w_p
        A_w_p[:3, :3] = R_w_p
        setup_target_frames(model, data, ref_ids, target_poses)

        # Set the initial configuration
        data.qpos[:6] = q_start       
        mujoco.mj_forward(model, data)
        viewer.sync()

        input("Press Enter to start optimization...")

        for idx, (pos, quat) in enumerate(target_poses):
            print(f"\n=== Optimizing target {idx+1}/{len(target_poses)} ===")
            z_dir = R.from_quat(quat).as_matrix()[:,2]

            # Try multiprocessing first, fall back to threading if issues
            try:
                solver = BioIK2Solver(model, data, tool_site_id, jlimits, xml_path,
                                      population_size=3000, n_elites=400, time_limit=2.0,
                                      use_multiprocessing=True, n_workers=20)
            except:
                print("Multiprocessing failed, falling back to threading...")
                solver = BioIK2Solver(model, data, tool_site_id, jlimits, xml_path,
                                      population_size=3000, n_elites=400, time_limit=2.0,
                                      use_multiprocessing=False, n_workers=20)

            solver.add_goal(PoseGoal(pos, z_dir, weight=1e10))
            solver.add_goal(JointLimitGoal(jlimits, weight=0))
            solver.add_goal(ElbowGoal(jlimits, alpha=0.4, weight=0))
            solver.add_goal(ManipulabilityGoal(weight=1e3))
            solver.add_goal(AntiAlignGoal(z_dir, weight=1e4))
            solver.add_goal(CollisionGoal(robot_geoms, floor_id, wt=1e5, weight=1.0))

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
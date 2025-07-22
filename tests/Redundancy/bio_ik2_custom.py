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

# Helper functions for visualization setup
def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def set_body_pose(model, data, body_id, pos, quat):
    model.body_pos[body_id] = pos
    model.body_quat[body_id] = quat
    mujoco.mj_forward(model, data)

def setup_target_frames(model, data, ref_body_ids, target_poses):
    for i, (pos, quat) in enumerate(target_poses):
        set_body_pose(model, data, ref_body_ids[i],
                      pos, [quat[3], quat[0], quat[1], quat[2]])
    mujoco.mj_forward(model, data)

# BioIK2 core solver and goal classes
class Goal:
    def __init__(self, weight=1.0): self.weight = weight
    def error(self, q, model, data, tool_site_id): raise NotImplementedError

class PoseGoal(Goal):
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
    def __init__(self, joint_lims, weight=1.0):
        super().__init__(weight)
        lims        = np.array(joint_lims)
        self.mid    = 0.5*(lims[:,0] + lims[:,1])
        self.range  = lims[:,1] - lims[:,0]
    def error(self, q, model, data, tool_site_id):
        d = 2*np.abs(q - self.mid) - 0.5*self.range
        return np.linalg.norm(d)

class ElbowGoal(Goal):
    def __init__(self, joint_lims, alpha=0.4, weight=1.0):
        super().__init__(weight)
        low, high = joint_lims[3]
        self.pref = low + alpha*(high - low)
    def error(self, q, model, data, tool_site_id):
        return abs(q[3] - self.pref)

class ManipulabilityGoal(Goal):
    def error(self, q, model, data, tool_site_id):
        data.qpos[:model.nv] = q; mujoco.mj_forward(model, data)
        Jp = np.zeros((3,model.nv)); Jr = np.zeros((3,model.nv))
        mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
        J = np.vstack([Jp, Jr])[:,:6]
        JJt = J @ J.T
        det = np.linalg.det(JJt)
        return 1e3 if det <= 1e-12 else 1.0/np.sqrt(det)

class AntiAlignGoal(Goal):
    def __init__(self, target_z, weight=1.0):
        super().__init__(weight)
        self.target_z = np.array(target_z)
    def error(self, q, model, data, tool_site_id):
        data.qpos[:model.nv] = q; mujoco.mj_forward(model, data)
        dot = np.dot(data.site_xmat[tool_site_id].reshape(3,3)[:,2], self.target_z)
        return max(0.0, -dot)

class CollisionGoal(Goal):
    def __init__(self, robot_geom_ids, floor_geom_id=None, wt=1e5, weight=1.0):
        super().__init__(weight)
        self.robot_geom_ids = robot_geom_ids
        self.floor_id       = floor_geom_id
        self.wt             = wt
    def error(self, q, model, data, tool_site_id):
        data.qpos[:model.nv] = q; mujoco.mj_forward(model, data); mujoco.mj_collision(model, data)
        pen = 0.0
        for i in range(data.ncon):
            c = data.contact[i]; g1, g2 = c.geom1, c.geom2
            rr = (g1 in self.robot_geom_ids and g2 in self.robot_geom_ids and g1!=g2)
            rf = (self.floor_id is not None and 
                  ((g1 in self.robot_geom_ids and g2==self.floor_id) or 
                   (g2 in self.robot_geom_ids and g1==self.floor_id)))
            if rr or rf:
                pen += max(0.0, -c.dist)*1000.0 + 1.0
        return self.wt * pen

class BioIK2Solver:
    def __init__(self, model, data, tool_site_id, joint_lims,
                 population_size=50, n_elites=5, time_limit=0.1):
        self.model, self.data, self.tool = model, data, tool_site_id
        lims = np.array(joint_lims)
        self.lb, self.ub = lims[:,0], lims[:,1]
        self.dim, self.pop_size, self.n_elites = len(self.lb), population_size, n_elites
        self.time_limit, self.goals, self.eval_count = time_limit, [], 0

    def add_goal(self, goal: Goal):
        self.goals.append(goal)

    def cost(self, q):
        self.eval_count += 1
        return np.sqrt(sum((g.weight * g.error(q, self.model, self.data, self.tool))**2 
                           for g in self.goals))

    def solve(self, tol=1e-2):
        pop    = [self.lb + np.random.rand(self.dim)*(self.ub-self.lb) for _ in range(self.pop_size)]
        costs  = [self.cost(q) for q in pop]
        start  = time.time()
        best   = min(costs)
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

            idx_sort = np.argsort(costs)
            pop   = [pop[i]   for i in idx_sort]
            costs = [costs[i] for i in idx_sort]
            best  = costs[0]

            # polish elites
            for i in range(self.n_elites):
                res = minimize(lambda x: self.cost(x), pop[i], 
                               method='L-BFGS-B', 
                               bounds=list(zip(self.lb, self.ub)), 
                               options={'ftol':1e-6})
                pop[i]   = res.x
                costs[i] = self.cost(res.x)
                best     = min(best, costs[i])

            # breed
            new_pop = pop[:self.n_elites]
            while len(new_pop) < self.pop_size:
                a, b = np.random.choice(self.pop_size, 2, replace=False)
                p1, p2 = pop[a], pop[b]
                alpha = np.random.rand(self.dim)
                child = alpha*p1 + (1-alpha)*p2
                mask  = np.random.rand(self.dim) < (1.0/self.dim)
                child[mask] += (np.random.rand(mask.sum())*2-1)*(self.ub-self.lb)[mask]
                new_pop.append(np.clip(child, self.lb, self.ub))

            pop   = new_pop
            costs = [self.cost(q) for q in pop]
            best  = min(best, min(costs))

        print(f"Completed solve(): iterations={iteration}, best_cost={best:.3e}, cost_evals={self.eval_count}")
        return pop[int(np.argmin(costs))]

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, "ur5e_utils_mujoco/scene.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    tool_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')
    base_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'tool_frame')
    jlimits      = model.jnt_range[:6]

    robot_geoms = [i for i in range(model.ngeom)
                   if 'collision' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')

    target_poses = [
        (np.array([0.2,-0.2,0.2]), R.from_euler('xyz',[180,0,0],True).as_quat()),
        (np.array([0.3, 0.1,0.7]), R.from_euler('xyz',[0,0,0],True).as_quat()),
        (np.array([0.3, 0.3,0.3]), R.from_euler('xyz',[135,0,90],True).as_quat()),
        (np.array([-0.4,0.5,0.7]), R.from_euler('xyz',[30,0,0],True).as_quat()),
        (np.array([0.2,0.2,0.1]), R.from_euler('xyz',[180,0,90],True).as_quat()),
    ]

    q_start = np.radians([-8.38,-68.05,-138,-64,90,-7.85])
    show_pose_duration = 3.0

    ref_ids = []
    for i in range(len(target_poses)):
        name = f'reference_target_{i+1}'
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'reference_target')
        ref_ids.append(bid)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        model.body_pos[base_id]  = [-0.1, -0.1, 0.15]
        model.body_quat[base_id] = euler_to_quaternion(45,0,0,True)
        model.body_pos[tool_body_id]  = [0.1, 0.1, 0.25]
        model.body_quat[tool_body_id] = euler_to_quaternion(0,0,0,True)

        data.qpos[:6] = q_start
        setup_target_frames(model, data, ref_ids, target_poses)
        mujoco.mj_forward(model, data)
        viewer.sync()

        input("Press Enter to start optimization...")

        for idx, (pos, quat) in enumerate(target_poses):
            print(f"\n=== Optimizing target {idx+1}/{len(target_poses)} ===")
            z_dir = R.from_quat(quat).as_matrix()[:,2]

            solver = BioIK2Solver(model, data, tool_id, jlimits,
                                  population_size=60, n_elites=7, time_limit=2.0)
            solver.add_goal(PoseGoal(pos, z_dir,     weight=1e10))
            solver.add_goal(JointLimitGoal(jlimits, weight=1e2))
            solver.add_goal(ElbowGoal(jlimits, alpha=0.4, weight=1e2))
            solver.add_goal(ManipulabilityGoal(weight=1e3))
            solver.add_goal(AntiAlignGoal(z_dir, weight=1e4))
            solver.add_goal(CollisionGoal(robot_geoms, floor_id, wt=1e5, weight=1.0))

            t0 = time.time()
            q_opt = solver.solve()
            print(f"Time: {time.time()-t0:.2f}s")

            data.qpos[:6] = q_opt
            mujoco.mj_forward(model, data)
            viewer.sync()

            time.sleep(show_pose_duration)
            viewer.sync()

        input("Optimization sequence completed. Press Enter to exit...")

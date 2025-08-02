#!/usr/bin/env python3
import sys, os
from pathlib import Path
import time
from contextlib import redirect_stderr
import contextlib

''' 
This code is an improvement of 'inference_single_pose.py' for two reasons:
    1. It allows to test the network on multiple poses in parallel, which is much faster.
    2. It allows to test the network on a rotational sweep of a base pose, which is useful for testing the task redundancy resolution.
'''

# Specify the path for ikflow
script_path  = Path(__file__).resolve()
project_root = script_path.parents[2]
ikflow_path  = project_root / "ikflow"
if not ikflow_path.exists():
    raise FileNotFoundError(f"Cannot find local ikflow at {ikflow_path}")
sys.path.pop(0)
sys.path.insert(0, str(ikflow_path))

# import
import torch
from jrl.robots import Robot

from ikflow.model           import IkflowModelParameters
from ikflow.ikflow_solver   import IKFlowSolver
from ikflow.training.lt_model import IkfLitModel
from ikflow.training.lt_data  import IkfLitDataset
from ikflow.config            import DATASET_TAG_NON_SELF_COLLIDING, DATASET_DIR
from ikflow.utils             import get_dataset_filepaths

@contextlib.contextmanager
def suppress_native_stderr():
    """
    Redirect the OS-level stderr (fd 2) into /dev/null (or NUL on Windows)
    so that even native extensions’ prints to stderr disappear.
    """
    # Open the null device
    devnull = os.open(os.devnull, os.O_RDWR)
    # Duplicate the current stderr fd (so we can restore it later)
    old_stderr = os.dup(2)
    # Replace stderr with our devnull
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        # Restore the original stderr
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)

def quat_to_mat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion(s) (w,x,y,z) → rotation matrix/matrices on GPU.
    """
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = quat.unbind(-1)
    xx = x*x; yy = y*y; zz = z*z; ww = w*w
    xy = x*y; xz = x*z; xw = x*w
    yz = y*z; yw = y*w; zw = z*w

    B = quat.shape[:-1]
    M = torch.empty(*B, 3, 3, device=quat.device, dtype=quat.dtype)
    M[...,0,0] = ww + xx - yy - zz
    M[...,0,1] = 2*(xy - zw)
    M[...,0,2] = 2*(xz + yw)
    M[...,1,0] = 2*(xy + zw)
    M[...,1,1] = ww - xx + yy - zz
    M[...,1,2] = 2*(yz - xw)
    M[...,2,0] = 2*(xz - yw)
    M[...,2,1] = 2*(yz + xw)
    M[...,2,2] = ww - xx - yy + zz
    return M

class FastIKFlowSolver:
    def __init__(self):
        self._setup_solver()

    def _setup_solver(self):
        # choose device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # robot
        urdf_path = project_root / "ur5e_utils_mujoco" / "ur5e" / "ur5e.urdf"
        with suppress_native_stderr():

            robot = Robot(
                name="ur5e_custom",
                urdf_filepath=str(urdf_path),
                active_joints=[
                    "shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
                    "wrist_1_joint","wrist_2_joint","wrist_3_joint",
                ],
                base_link="base_link",
                end_effector_link_name="wrist_3_link",
                ignored_collision_pairs=[],
                collision_capsules_by_link=None,
            )

        # hyper-parameters
        h = IkflowModelParameters()
        h.coupling_layer         = "glow"
        h.nb_nodes               = 6
        h.dim_latent_space       = 8
        h.coeff_fn_config        = 3
        h.coeff_fn_internal_size = 1024
        h.rnvp_clamp             = 2.5
        h.softflow_enabled       = True
        h.softflow_noise_scale   = 0.001
        h.y_noise_scale          = 1e-7
        h.zeros_noise_scale      = 1e-3
        h.sigmoid_on_output      = False

        # instantiate solver
        self.solver = IKFlowSolver(h, robot)

        # load dataset transforms
        suffix = f"_{DATASET_TAG_NON_SELF_COLLIDING}"
        ddir   = os.path.join(DATASET_DIR, f"{robot.name}{suffix}")
        samples_fp, poses_fp, *_ = get_dataset_filepaths(ddir, [DATASET_TAG_NON_SELF_COLLIDING])
        samples_tr = torch.load(samples_fp).float().to(self.device)
        poses_tr   = torch.load(poses_fp).float().to(self.device)
        x_mean, x_std = samples_tr.mean(0), samples_tr.std(0)
        reorder = torch.tensor([0,1,2,6,3,4,5], device=self.device)
        poses_re = poses_tr[:, reorder]
        y_mean, y_std = poses_re.mean(0), poses_re.std(0)

        ds = IkfLitDataset(
            robot_name   = robot.name,
            batch_size   = 1,
            val_set_size = 1,
            dataset_tags = [DATASET_TAG_NON_SELF_COLLIDING],
        )
        ds.prepare_data(); ds.setup(stage="fit")
        self.solver._x_transform = ds._x_transform
        self.solver._y_transform = ds._y_transform
        self.solver._x_transform.mean, self.solver._x_transform.std = x_mean, x_std
        self.solver._y_transform.mean, self.solver._y_transform.std = y_mean, y_std

        # load Lightning checkpoint
        '''
        ckpt = (
            P.home() / ".cache" / "ikflow" / "training_logs" /
            "ur5e_custom--Jul.21.2025_08-43AM" /
            "ikflow-checkpoint-epoch-epoch=199.ckpt"
        )
        '''
        ckpt = (
            project_root
            / "ikflow" / "ikflow" / "weights"
            / "ur5e_custom--Jul.22.2025_06-17PM"
            / "ikflow-checkpoint-epoch-epoch=216.ckpt"
        )
        lit = IkfLitModel.load_from_checkpoint(
            str(ckpt),
            ik_solver        = self.solver,
            base_hparams     = h,
            learning_rate    = 1e-4,
            checkpoint_every = 0,
            log_every        = 0,
            gradient_clip    = 1.0,
            gamma            = 0.9794578299341784,
            step_lr_every    = int(2.5e6/64),
            weight_decay     = 1.8e-05,
            optimizer_name   = "adamw",
            sigmoid_on_output= False,
            strict           = True,
        )
        lit.to(self.device).eval()
        for p in lit.parameters(): p.requires_grad = False

        # tell solver that weights are loaded
        self.solver._model_weights_loaded = True

        # compile for speed
        try:
            self.solver = torch.compile(self.solver, mode="max-autotune") #! pick the fastest kernel configuration
        except:
            pass

    def solve_ik_ultra_fast(self, target_pose: torch.Tensor, N: int = 1000):
        target = target_pose.to(self.device).view(1,7)
        with torch.no_grad():
            sols_ex, valid = self.solver.generate_exact_ik_solutions(
                target.expand(N,7),
                pos_error_threshold=2e-3,
                rot_error_threshold=2e-2,
                repeat_counts=(1,2),
            )
        valid = valid.cpu()
        sols_ex = sols_ex.cpu()
        val_sols = sols_ex[valid]
        if val_sols.numel():
            fk = torch.tensor(
                self.solver.robot.forward_kinematics_klampt(val_sols.numpy()),
                dtype=torch.float32
            )
        else:
            fk = torch.empty(0,7)
        return val_sols, fk
    
    def solve_rotational_sweep(self, base_pose: torch.Tensor, N_samples: int = 500, N_disc: int = 18):
        """
        Sequential sweep: each of the N_disc poses solved independently
        with N_samples each.
        """
        base = base_pose.to(self.device).view(7)
        pos, quat = base[:3], base[3:]

        # build Z-axis rotation quaternions
        angles = torch.linspace(0, 2*torch.pi, N_disc, device=self.device)
        half = angles * 0.5 #! q = (cos(θ/2), sin(θ/2) * u) => I need θ/2
        qz = torch.stack([
            torch.cos(half),
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half),
        ], dim=1) 

        # ! Incrementally rotate around z
        results = []
        for i in range(N_disc):
            # ! q_start * q_z
            w1,x1,y1,z1 = quat 
            w2,x2,y2,z2 = qz[i]
            qr = torch.tensor([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ], device=self.device)

            pose_i = torch.cat([pos, qr], dim=0)
            sols_i, fk_i = self.solve_ik_ultra_fast(pose_i, N_samples)
            results.append((sols_i, fk_i))

        return results

# Additional methods
def solve_ik_fast(target_pose: torch.Tensor, N: int = 1000, fast_solver: FastIKFlowSolver = None):
    return fast_solver.solve_ik_ultra_fast(target_pose, N)

def solve_ik_rotational_sweep(base_pose: torch.Tensor, N_samples: int = 5000, N_disc: int = 360, fast_solver: FastIKFlowSolver = None):
    return fast_solver.solve_rotational_sweep(base_pose, N_samples, N_disc)

# Test the code
if __name__ == "__main__":

    # Define an instance of FastIKFlowSolver
    fast_ik_solver = FastIKFlowSolver()

    # Define a target pose
    target = torch.tensor([
        -0.5430, -0.0486,  0.4806,
         0.3760, -0.5168,  0.4190,  0.6450
    ])
    print("=== Single Pose Test ===")
    t0 = time.time()
    sols, fk = solve_ik_fast(target, N=1000, fast_solver=fast_ik_solver)
    print(f"Found {len(sols)} solutions in {(time.time()-t0)*1000:.1f} ms")

    # Solve task redundancy
    print("\n Resolution of the task redundancy")
    t1 = time.time()
    sweeps = solve_ik_rotational_sweep(target, N_samples=200, N_disc=72, fast_solver=fast_ik_solver)
    dt = time.time() - t1
    total = sum(len(s) for s,_ in sweeps)  
    print(f"Total solutions: {total}; Time: {dt:.3f}s; Avg: {1000*dt/72:.2f}ms/pose")

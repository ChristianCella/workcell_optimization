#!/usr/bin/env python3
import sys, os
from pathlib import Path

''' 
This code allows you to test the training of ikflow. Basically it is similar to the script 'inference.py' in 'ikflow',
but it allows to test the network directly in this directory.
The function defined here is used also in 'impose_q.py', in the case that you do not want to test a single hard-coded configuration.
'''

# Specify the path for ikflow
script_path  = Path(__file__).resolve()
project_root = script_path.parents[3]
ikflow_path  = project_root / "ikflow"
if not ikflow_path.exists():
    raise FileNotFoundError(f"Cannot find local ikflow at {ikflow_path}")
sys.path.pop(0)
sys.path.insert(0, str(ikflow_path))

# Imports
import torch
from pathlib import Path as P
from jrl.robots import Robot

from ikflow.model           import IkflowModelParameters
from ikflow.ikflow_solver   import IKFlowSolver
from ikflow.training.lt_model import IkfLitModel
from ikflow.training.lt_data  import IkfLitDataset
from ikflow.config            import DATASET_TAG_NON_SELF_COLLIDING, DATASET_DIR
from ikflow.utils             import get_dataset_filepaths

# Build a loaded solver
def setup_solver():

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # robot
    urdf_path = project_root / "ur5e_utils_mujoco" / "ur5e.urdf"
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

    # hyperparameters (same as those used in training)
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

    # solver skeleton
    solver = IKFlowSolver(h, robot)

    # load dataset means/stds and normalize
    suffix = f"_{DATASET_TAG_NON_SELF_COLLIDING}"
    ddir   = os.path.join(DATASET_DIR, f"{robot.name}{suffix}")
    samples_fp, poses_fp, *_ = get_dataset_filepaths(ddir, [DATASET_TAG_NON_SELF_COLLIDING])
    samples_tr = torch.load(samples_fp).float().to(device)
    poses_tr   = torch.load(poses_fp).float().to(device)

    # normalize (NOTE: the network was trained with this, so it's paramount to make inference with this)
    x_mean = samples_tr.mean(dim=0)
    x_std  = samples_tr.std(dim=0)
    reorder  = torch.tensor([0,1,2,6,3,4,5], device=device)
    poses_re = poses_tr[:, reorder]
    y_mean   = poses_re.mean(dim=0)
    y_std    = poses_re.std(dim=0)

    # attach transforms
    ds = IkfLitDataset(
        robot_name   = robot.name,
        batch_size   = 1,
        val_set_size = 1,
        dataset_tags = [DATASET_TAG_NON_SELF_COLLIDING],
    )
    ds.prepare_data()
    ds.setup(stage="fit")
    solver._x_transform = ds._x_transform
    solver._y_transform = ds._y_transform
    solver._x_transform.mean = x_mean
    solver._x_transform.std  = x_std
    solver._y_transform.mean = y_mean
    solver._y_transform.std  = y_std

    # Load weights (after training)
    #! Old (working nicely) weights
    '''
    ckpt = (
        P.home() / ".cache" / "ikflow" / "training_logs" /
        "ur5e_custom--Jul.21.2025_08-43AM" /
        "ikflow-checkpoint-epoch-epoch=199.ckpt"
    )
    '''
    #! New weights
    ckpt = (
        project_root
        / "ikflow"  # the top‐level ikflow repo folder
        / "ikflow"  # the python package subfolder
        / "weights"
        / "ur5e_custom--Jul.22.2025_06-17PM"
        / "ikflow-checkpoint-epoch-epoch=214.ckpt"
    )
    lit = IkfLitModel.load_from_checkpoint(
        str(ckpt),
        ik_solver        = solver,
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
    lit.to(device).eval()
    solver._model_weights_loaded = True

    return solver, device


# most important function
def solve_ik(target_pose: torch.Tensor, N: int = 50):
    """
    Given target_pose (7,) = (x,y,z,qw,qx,qy,qz) and sample count N,
    returns:
      sols_ok: Tensor[M,6]  -- refined joint solutions
      fk_ok:   Tensor[M,7]  -- exact FK poses for those sols
    where M ≤ N are only the successful refinements.
    """
    solver, device = setup_solver()
    tgt = target_pose.to(device).view(1,7)

    # approximate + small noise (NOTE: usually the approx. solution is already good enough, but not perfect)
    sols, *_ = solver.generate_ik_solutions(
        tgt, n=N, latent_scale=0.01,
        clamp_to_joint_limits=True,
        return_detailed=True
    )
    sols = sols.cpu()

    # exact Newton refine
    sols_ex, valid = solver.generate_exact_ik_solutions(
        tgt.expand(N,7),
        pos_error_threshold=1e-3,
        rot_error_threshold=1e-2,
        repeat_counts=(1,3,10),
    )
    sols_ex = sols_ex.cpu()
    valid  = valid.cpu()

    # get FK poses
    fk_all = torch.tensor(
        solver.robot.forward_kinematics_klampt(sols_ex.numpy()),
        dtype=torch.float32
    )

    # filter only the OK ones (NOTE: if n_epochs in training increases, this will return more solutions)
    sols_ok = sols_ex[valid]
    fk_ok   = fk_all[valid]

    return sols_ok, fk_ok


if __name__ == "__main__":

    # target pose
    target = torch.tensor([
        -0.5430, -0.0486,  0.4806,
         0.3760, -0.5168,  0.4190,  0.6450
    ])

    sols, fk = solve_ik(target, N=50)
    print(f"\nFound {sols.shape[0]} valid IK solutions:\n")
    for i, (q, x) in enumerate(zip(sols, fk), 1):
        print(f" #{i:2d} q = {q.tolist()}")
        print(f"      x = {x.tolist()}\n")

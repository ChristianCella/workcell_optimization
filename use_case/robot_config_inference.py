#!/usr/bin/env python3
import sys, os
from pathlib import Path

# ─── A) Make sure we import your editable ikflow clone first ───────────────
script_path  = Path(__file__).resolve()
project_root = script_path.parents[2]       # use_case → workcell_optimization → Robotic_contact_operations
ikflow_path  = project_root / "ikflow"
if not ikflow_path.exists():
    raise FileNotFoundError(f"Cannot find local ikflow at {ikflow_path}")
sys.path.pop(0)
sys.path.insert(0, str(ikflow_path))

# ─── B) Import config first for its startup prints ──────────────────────────
#import ikflow.config  # prints the device & dataset‐storage lines

# ─── C) Now bring in the rest ───────────────────────────────────────────────
import torch
from jrl.robots import Robot

from ikflow.model           import IkflowModelParameters
from ikflow.ikflow_solver   import IKFlowSolver
from ikflow.training.lt_model import IkfLitModel
from ikflow.training.lt_data  import IkfLitDataset
from ikflow.config            import DATASET_TAG_NON_SELF_COLLIDING, DATASET_DIR
from ikflow.utils             import get_dataset_filepaths

def main():
    # ─── 0) Device ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # ─── 1) Instantiate your UR5e ───────────────────────────────────────────
    urdf_path = project_root / "ur5e_utils_mujoco" / "ur5e.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found at {urdf_path}")
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
    print(f"Loaded robot '{robot.name}' (DOF={robot.ndof})\n")

    # ─── 2) Recreate your training hyper-parameters exactly ─────────────────
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

    # ─── 3) Build an empty IKFlow solver ─────────────────────────────────
    solver = IKFlowSolver(h, robot)

    # ─── 4) Load dataset means/stds and hard-wire them ────────────────────
    suffix = f"_{DATASET_TAG_NON_SELF_COLLIDING}"
    ddir   = os.path.join(DATASET_DIR, f"{robot.name}{suffix}")
    samples_fp, poses_fp, *_ = get_dataset_filepaths(ddir, [DATASET_TAG_NON_SELF_COLLIDING])
    print("Loading training set from:\n ", samples_fp, "\n ", poses_fp)
    samples_tr = torch.load(samples_fp).float().to(device)
    poses_tr   = torch.load(poses_fp).float().to(device)

    # compute joint‐space normalization
    x_mean = samples_tr.mean(dim=0)
    x_std  = samples_tr.std(dim=0)

    # reorder quaternion to (qw,qx,qy,qz)
    reorder  = torch.tensor([0,1,2,6,3,4,5], device=device)
    poses_re = poses_tr[:, reorder]
    y_mean   = poses_re.mean(dim=0)
    y_std    = poses_re.std(dim=0)

    print("→ joint mean:", x_mean.cpu().numpy())
    print("→ joint std: ", x_std.cpu().numpy())
    print("→ pose mean:", y_mean.cpu().numpy())
    print("→ pose std: ", y_std.cpu().numpy(), "\n")

    # ─── 5) Attach the SAME transforms used in training ────────────────────
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

    # ─── 6) Load your Lightning checkpoint ────────────────────────────────
    ckpt = (
        Path.home() / ".cache" / "ikflow" / "training_logs" /
        "ur5e_custom--Jul.21.2025_08-43AM" /
        "ikflow-checkpoint-epoch-epoch=199.ckpt"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found")
    print("Loading checkpoint:", ckpt.name)
    lit = IkfLitModel.load_from_checkpoint(
        str(ckpt),
        ik_solver        = solver,
        base_hparams     = h,
        learning_rate    = 1e-4,
        checkpoint_every = 250000,
        log_every        = 20000,
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
    print("Model weights loaded into IKFlowSolver.\n")

    # ─── 7) SANITY‐CHECK on the first 5 training poses (MAP only) ─────────
    print("=== SANITY CHECK ON TRAINING ENDPOINTS (MAP) ===")
    first5 = poses_re[:5]
    sols_tr, perr_tr, rerr_tr, *_ = solver.generate_ik_solutions(
        first5,
        n                 = first5.shape[0],
        latent_scale      = 1e-6,    # tiny >0 so we don't trip assert
        clamp_to_joint_limits=True,
        return_detailed   = True,
    )
    print(" train pos err:", perr_tr.cpu().numpy())
    print(" train rot err:", rerr_tr.cpu().numpy(), "\n")

    # ─── 8) Now your real target pose ──────────────────────────────────────
    x,y,z,qw,qx,qy,qz = -0.5430, -0.0486,  0.4806,  0.3760, -0.5168,  0.4190,  0.6450
    target = torch.tensor([x,y,z, qw,qx,qy,qz], device=device)
    print("Target pose (7D):", target.cpu().tolist(), "\n")

    # ─── 9) Approximate IK (MAP + small latent noise) ─────────────────────
    N = 50
    sols, perr, rerr, jl, sc, t = solver.generate_ik_solutions(
        target,
        n                 = N,
        latent_scale      = 0.01,
        clamp_to_joint_limits=True,
        return_detailed   = True,
    )
    sols = sols.cpu()
    #print(f"Approximate IK (MAP+noise) in {t*1000:.1f}ms")
    #print(" pos err:", perr.cpu().numpy())
    #print(" rot err:", rerr.cpu().numpy())
    #print(" jl flags:", jl.cpu().numpy())
    #print(" sc flags:", sc.cpu().numpy(), "\n")
    #for i, q in enumerate(sols, 1):
        #print(f"  Solution {i:2d}: {q.tolist()}")
    #print()

    # ───10) Exact Newton refinement ────────────────────────────────────────
    sols_ex, valid = solver.generate_exact_ik_solutions(
        target.expand(N,7),
        pos_error_threshold=1e-3,
        rot_error_threshold=1e-2,
        repeat_counts      =(1,3,10),
    )
    sols_ex = sols_ex.cpu()
    #print("Exact-refined valid flags:", valid.cpu().numpy(), "\n")
    for i, (q, ok) in enumerate(zip(sols_ex, valid.cpu()), 1):
        print(f"  [{'OK' if ok else 'BAD'}] {i:2d}: {q.tolist()}")
    print()

    # ───11) Final FK check ────────────────────────────────────────────────
    #print("=== FK of approx ===")
    #for i, (x,y,z,qw,qx,qy,qz) in enumerate(robot.forward_kinematics_klampt(sols.numpy()), 1):
        #print(f"{i:2d}: x={x:.3f}, y={y:.3f}, z={z:.3f}, qw={qw:.3f}, qx={qx:.3f}, qy={qy:.3f}, qz={qz:.3f}")
    print("\n=== FK of exact ===")
    for i, (x,y,z,qw,qx,qy,qz) in enumerate(robot.forward_kinematics_klampt(sols_ex.numpy()), 1):
        print(f"{i:2d}: x={x:.3f}, y={y:.3f}, z={z:.3f}, qw={qw:.3f}, qx={qx:.3f}, qy={qy:.3f}, qz={qz:.3f}")
    print()

def _datetime_str():
    from datetime import datetime
    return datetime.now().strftime("%b.%d.%Y_%I-%M%p")

if __name__ == "__main__":
    main()

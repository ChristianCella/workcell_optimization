import os, sys
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from scipy.interpolate import interp1d

#* Base directrory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def densify_cartesian_path(cartesian_path, eef_step=0.01):
    """
    cartesian_path: list of (position, euler_angles) tuples
                    where position is (3,) and euler_angles is (3,) in radians
    eef_step      : interpolation resolution in meters
    """
    positions    = np.array([p for p, _ in cartesian_path])   # (N, 3)
    euler_angles = np.array([e for _, e in cartesian_path])   # (N, 3)

    # Convert euler → quaternions [x, y, z, w]
    quaternions = Rotation.from_euler('xyz', euler_angles).as_quat()

    # Compute cumulative arc length along the path
    dists = [0.0]
    for i in range(1, len(positions)):
        d = np.linalg.norm(positions[i] - positions[i-1])
        dists.append(dists[-1] + d)
    total = dists[-1]

    # New sample times in [0, 1]
    n_points = max(2, int(total / eef_step))
    t_orig = np.array(dists) / total
    t_new  = np.linspace(0, 1, n_points)

    # Interpolate positions linearly
    pos_interp       = interp1d(t_orig, positions, axis=0)
    positions_dense  = pos_interp(t_new)

    # Interpolate orientations with SLERP
    rotations          = Rotation.from_quat(quaternions)
    slerp              = Slerp(t_orig, rotations)
    euler_dense        = slerp(t_new).as_euler('xyz')          # back to euler

    # Rebuild in your original format: list of (position, euler_angles)
    cartesian_path_dense = [
        (positions_dense[i], euler_dense[i])
        for i in range(n_points)
    ]

    print(f"Densified path: {len(cartesian_path)} → {n_points} waypoints "
          f"(total length: {total:.3f} m, step: {eef_step} m)")

    return cartesian_path_dense
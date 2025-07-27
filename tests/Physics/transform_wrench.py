import numpy as np
import sys, os
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from transformations import get_world_wrench

if __name__ == "__main__":

    verbose = True
    check = False

    # External wrench in the local frame
    wrench_local = np.array([0, 0, -30, 0, 0, -10])  # Fx, Fy, Fz, Mx, My, Mz

    # Local frame with respect to the world frame
    euler_angles = np.radians([180, 45, 0])
    R_l_w = R.from_euler('XYZ', euler_angles).as_matrix() #! Remember: specify 'XYZ' to work with intrinsic rotations
    if verbose: print("Local frame rotation matrix with respect to the world frame:\n", R_l_w)

    # NOTE: check the rotation matrix
    rx = R.from_euler('x', 180, degrees=True)
    ry = R.from_euler('y', 45, degrees=True)
    rz = R.from_euler('z', 0, degrees=True)
    R_l_w_check = rx * ry * rz
    if verbose and check: print("Correct rotation matrix from local to world frame:\n", R_l_w_check.as_matrix())

    # Wrench in the world frame
    F_world = get_world_wrench(R_l_w, wrench_local)  # This function should return the wrench in the world frame
    if verbose: print("Force in world frame:", F_world)


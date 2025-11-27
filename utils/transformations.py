import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler('XYZ', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def quaternion_to_euler(q, degrees=False):
    from scipy.spatial.transform import Rotation as R

    # Convert [w, x, y, z] → [x, y, z, w] for scipy
    quat_scipy = [q[1], q[2], q[3], q[0]]

    r = R.from_quat(quat_scipy)
    roll, pitch, yaw = r.as_euler('XYZ', degrees=degrees)

    return roll, pitch, yaw


def rotm_to_quaternion(rotm):
    from scipy.spatial.transform import Rotation as R
    q = R.from_matrix(rotm).as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def get_world_wrench(R, w_l):
    R_full = np.zeros((6, 6))
    R_full[:3, :3] = R
    R_full[3:, 3:] = R
    return R_full @ w_l

def get_homogeneous_matrix(tx, ty, tz, rx, ry, rz):
    t_vec = np.array([tx, ty, tz])
    R_mat = R.from_euler('XYZ', [np.radians(rx), np.radians(ry), np.radians(rz)], degrees=False).as_matrix()
    A_mat = np.eye(4)
    A_mat[:3, 3] = t_vec
    A_mat[:3, :3] = R_mat
    return t_vec, R_mat, A_mat

# Test the methods
if __name__ == "__main__":

    #* Test the passage from Euler angles to quaternion
    theta_x = -90
    theta_y = 76
    theta_z = 12
    q = euler_to_quaternion(theta_x, theta_y, theta_z, degrees=True)
    print("Quaternion: ", q)

    #* Test the passage from quaternion to Euler angles
    q = [-0.0226209, 0.978332,-0.0296377,-0.203658]
    roll, pitch, yaw = quaternion_to_euler(q, degrees=True)
    print(f"Euler angles: Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

    # External wrench in the local frame of the target
    wrench_local = np.array([0, 0, -30, 0, 0, -10])  # Fx, Fy, Fz, Mx, My, Mz

    # Local frame with respect to the world frame
    euler_angles = np.radians([180, 45, 0])
    R_l_w = R.from_euler('XYZ', euler_angles).as_matrix() #! Remember: specify 'XYZ' to work with intrinsic rotations
    print("\nLocal rotation, using scipy method:\n", R_l_w)

    # NOTE: check the rotation matrix
    rx = R.from_euler('x', 180, degrees=True)
    ry = R.from_euler('y', 45, degrees=True)
    rz = R.from_euler('z', 0, degrees=True)
    R_l_w_check = rx * ry * rz
    print("\nLocal rotation, computed by hand:\n", R_l_w_check.as_matrix())

    # Wrench in the world frame
    F_world = get_world_wrench(R_l_w, wrench_local)  # This function returns the wrench in the world frame
    print("\nForce in world frame:", F_world)
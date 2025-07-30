import numpy as np

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler('XYZ', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def rotm_to_quaternion(rotm):
    from scipy.spatial.transform import Rotation as R
    q = R.from_matrix(rotm).as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def get_world_wrench(R, w_l):
    R_full = np.zeros((6, 6))
    R_full[:3, :3] = R
    R_full[3:, 3:] = R
    return R_full @ w_l

if __name__ == "__main__":
    theta_x = 180
    theta_y = 0
    theta_z = 0
    q = euler_to_quaternion(theta_x, theta_y, theta_z, degrees=True)
    print("Quaternion:", q)
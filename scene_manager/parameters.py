from dataclasses import dataclass, field
import numpy as np

@dataclass
class TestIK:
    verbose: bool = True
    N_samples: int = 500  # 150
    N_disc: int = 1  # 60
    show_pose_duration: int = 0.01  # Seconds to show each pose

@dataclass
class Ur5eRobot:
    nu: int = 6 # Number of joints
    freq: int = 500 # Hz
    home_configuration: np.ndarray = field(default_factory=lambda: np.radians([90, -90, 90, -90, -90, 0]))
    gear_ratios: np.ndarray = field(default_factory=lambda: np.array([100, 100, 100, 100, 100, 100]))
    max_torques: np.ndarray = field(default_factory=lambda: np.array([1.50, 1.50, 1.50, 0.28, 0.28, 0.28]))
    robot_reach: float = 0.85 
    lb: list = field(default_factory=lambda: -2 * np.pi * np.ones(6))
    ub: list = field(default_factory=lambda: 2 * np.pi * np.ones(6))
    q_dot_max: np.ndarray = field(default_factory=lambda: np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14]))  # rad/s
    q_ddot_max: np.ndarray = field(default_factory=lambda: np.array([5.1, 5.1, 5.1, 5.1, 5.1, 5.1])) # rad/s²

@dataclass
class GoFaRobot:
    nu: int = 6 # Number of joints
    freq: int = 250 # Hz
    home_configuration: np.ndarray = field(default_factory=lambda: np.radians([0.0, 5.0, 4.0, 0.0, 80.0, 0.0]))
    robot_reach: float = 0.95 
    lb: list = field(default_factory=lambda: -2 * np.pi * np.ones(6))
    ub: list = field(default_factory=lambda: 2 * np.pi * np.ones(6))
    q_dot_max: np.ndarray = field(default_factory=lambda: np.array([2.18, 2.18, 2.44, 3.49, 3.49, 3.49]))
    q_ddot_max: np.ndarray = field(default_factory=lambda: np.array([3.1, 3.1, 3.1, 3.1, 3.1, 3.1])) 

@dataclass 
class FanucCrx10iaLRobot:
    nu: int = 6 # Number of joints
    freq: int = 25 # Hz
    home_configuration: np.ndarray = field(default_factory=lambda: np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    robot_reach: float = 1.418 
    lb: list = field(default_factory=lambda: np.radians([-360.0, -360.0, -540.0, -360.0, -360.0, -360.0]))
    ub: list = field(default_factory=lambda: np.radians([360.0, 360.0, 540.0, 360.0, 360.0, 360.0]))
    q_dot_max: np.ndarray = field(default_factory=lambda: np.radians([120.0, 120.0, 180.0, 180.0, 180.0, 180.0]))  # rad/s
    q_ddot_max: np.ndarray = field(default_factory=lambda: np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])) #! Not so sure

@dataclass 
class DoosanA0509Robot: #! CHECK PARAMETERS !!!
    nu: int = 6 # Number of joints
    freq: int = 25 # Hz
    home_configuration: np.ndarray = field(default_factory=lambda: np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    robot_reach: float = 1.418 
    lb: list = field(default_factory=lambda: np.radians([-360.0, -360.0, -540.0, -360.0, -360.0, -360.0]))
    ub: list = field(default_factory=lambda: np.radians([360.0, 360.0, 540.0, 360.0, 360.0, 360.0]))
    q_dot_max: np.ndarray = field(default_factory=lambda: np.radians([120.0, 120.0, 180.0, 180.0, 180.0, 180.0]))  # rad/s
    q_ddot_max: np.ndarray = field(default_factory=lambda: np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])) #! Not so sure





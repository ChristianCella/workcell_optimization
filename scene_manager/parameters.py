from dataclasses import dataclass, field
import numpy as np

@dataclass
class TestIK:
    verbose: bool = True
    N_samples: int = 100  # 150
    N_disc: int = 50  # 60
    show_pose_duration: int = 0.05  # Seconds to show each pose

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
    q_ddot_max: np.ndarray = field(default_factory=lambda: np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])) # rad/s²

@dataclass
class GoFaRobot:
    nu: int = 6 # Number of joints
    freq: int = 100 # Hz
    home_configuration: np.ndarray = field(default_factory=lambda: np.radians([0.0, 5.0, 4.0, 0.0, 80.0, 0.0]))
    robot_reach: float = 0.85 
    lb: list = field(default_factory=lambda: -2 * np.pi * np.ones(6))
    ub: list = field(default_factory=lambda: 2 * np.pi * np.ones(6))
    q_dot_max: np.ndarray = field(default_factory=lambda: np.array([2.18, 2.18, 2.4, 3.49, 3.49, 3.49]))  # rad/s
    q_ddot_max: np.ndarray = field(default_factory=lambda: np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])) # rad/s²

@dataclass #! Parameters to check!
class FanucCrx10iaLRobot:
    nu: int = 6 # Number of joints
    freq: int = 100 # Hz
    home_configuration: np.ndarray = field(default_factory=lambda: np.radians([0.0, 5.0, 4.0, 0.0, 80.0, 0.0]))
    robot_reach: float = 0.85 
    lb: list = field(default_factory=lambda: -2 * np.pi * np.ones(6))
    ub: list = field(default_factory=lambda: 2 * np.pi * np.ones(6))
    q_dot_max: np.ndarray = field(default_factory=lambda: np.array([2.18, 2.18, 2.4, 3.49, 3.49, 3.49]))  # rad/s
    q_ddot_max: np.ndarray = field(default_factory=lambda: np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])) # rad/s²





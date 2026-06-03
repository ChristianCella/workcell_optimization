from dataclasses import dataclass, field
import numpy as np

@dataclass
class TestIK:
    verbose: bool = True
    N_samples: int = 20  # Samples per 'discretized' pose configuration
    N_disc: int = 90  # Number of discrete configurations to test (rotational sweep)
    show_pose_duration: int = 0.5  # Seconds to show each pose

@dataclass
class Ur5eRobot:
    nu: int = 6 # Number of joints
    home_configuration: np.ndarray = field(default_factory=lambda: np.radians([90, -90, 90, -90, -90, 0]))
    gear_ratios: np.ndarray = field(default_factory=lambda: np.array([100, 100, 100, 100, 100, 100]))
    max_torques: np.ndarray = field(default_factory=lambda: np.array([1.50, 1.50, 1.50, 0.28, 0.28, 0.28]))
    robot_reach: float = 0.85 
    lb: list = field(default_factory=lambda: -2 * np.pi * np.ones(6))
    ub: list = field(default_factory=lambda: 2 * np.pi * np.ones(6))

@dataclass
class GoFaRobot:
    nu: int = 6 # Number of joints
    home_configuration: np.ndarray = field(default_factory=lambda: np.radians([0.0, 5.0, 4.0, 0.0, 80.0, 0.0]))
    gear_ratios: np.ndarray = field(default_factory=lambda: np.array([100, 100, 100, 100, 100, 100]))
    max_torques: np.ndarray = field(default_factory=lambda: np.array([1.50, 1.50, 1.50, 0.28, 0.28, 0.28]))
    robot_reach: float = 0.85 
    lb: list = field(default_factory=lambda: -2 * np.pi * np.ones(6))
    ub: list = field(default_factory=lambda: 2 * np.pi * np.ones(6))




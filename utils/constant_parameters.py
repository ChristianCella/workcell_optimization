from dataclasses import dataclass, field
import numpy as np

@dataclass
class TestIkFlow:
    verbose: bool = True
    N_samples: int = 30  # Samples per 'discretized' pose configuration
    N_disc: int = 10  # Number of discrete configurations to test (rotational sweep)
    show_pose_duration: int = 1  # Seconds to show each pose
    use_database: bool = True # Use txt database
    n_targets: int = 1  # Number of target reference frames
    x_tar: float = 0.4  # Target x position
    y_tar: float = 0.0  # Target y position
    z_tar: float = 0.4  # Target z position
    theta_x_tar : float = 180.0 # Target x orientation (deg)
    theta_y_tar : float = 0.0   # Target y orientation (deg)
    theta_z_tar : float = 0.0   # Target z orientation (deg)
    hande_offset : float = 0.157  # Length of the gripper hande
    extension_offset : float = 0.2  # Length of the extension tool

@dataclass
class OptimizationParameters:

    # Control variables
    verbose: bool = False # Display messages
    show_pose_duration: int = 0.05  # Seconds to show each pose
    activate_gui : bool = False  # Activate the GUI for visualization
    csv_directory: str = "screwing/turbo_ikflow"  # Directory to save CSV files

    # Ikflow variables
    N_samples: int = 25  # Samples per 'discretized' pose configuration
    N_disc: int = 90  # Number of discrete configurations to test (rotational sweep)
    
    # cma-es variables
    x0: np.ndarray = field(default_factory=lambda: 
                           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.radians(180), np.radians(-100), 
                                     np.radians(80), np.radians(-90), np.radians(-90), np.radians(45)]))  # initial mean mu
    sigma0 : float = 2  # initial std sigma
    popsize: int = 40  # number of individuals
    n_iter: int = 100  # number of iterations

@dataclass
class Ur5eRobot:
    nu: int = 6 # Number of joints
    gear_ratios: np.ndarray = field(default_factory=lambda: np.array([100, 100, 100, 100, 100, 100]))
    max_torques: np.ndarray = field(default_factory=lambda: np.array([1.50, 1.50, 1.50, 0.28, 0.28, 0.28])) # Those on the motors (not the joints)
    robot_reach: float = 0.85 # Radius of the maximum circle
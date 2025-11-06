from dataclasses import dataclass, field
import numpy as np

@dataclass
class TestIkFlow:
    verbose: bool = True
    N_samples: int = 30  # Samples per 'discretized' pose configuration
    N_disc: int = 10  # Number of discrete configurations to test (rotational sweep)
    use_ikflow: bool = True  # Set to False to test a hard-coded joint configuration
    show_pose_duration: int = 0.5  # Seconds to show each pose

@dataclass
class VisualRedundancy:
    verbose: bool = True
    N_samples: int = 50  # Samples per 'discretized' pose configuration
    N_disc: int = 8  # Number of discrete configurations to test (rotational sweep)
    use_ikflow: bool = True  # Set to False to test a hard-coded joint configuration
    show_pose_duration: int = 0.1  # Seconds to show each pose

@dataclass
class ScrewingTurbo:

    # Control variables
    verbose: bool = False # Display messages
    show_pose_duration: int = 0.05  # Seconds to show each pose
    activate_gui : bool = True  # Activate the GUI for visualization
    csv_directory: str = "optimization"  # Directory to save CSV files

    # Ikflow variables
    N_samples: int = 5  # Samples per 'discretized' pose configuration
    N_disc: int = 2  # Number of discrete configurations to test (rotational sweep)
    
    # cma-es variables
    x0: np.ndarray = field(default_factory=lambda: 
                           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.radians(180), np.radians(-100), 
                                     np.radians(80), np.radians(-90), np.radians(-90), np.radians(45)]))  # initial mean mu
    sigma0 : float = 2  # initial std sigma
    popsize: int = 2  # number of individuals
    n_iter: int = 2  # number of iterations

@dataclass
class Ur5eRobot:
    nu: int = 6 # Number of joints
    gear_ratios: np.ndarray = field(default_factory=lambda: np.array([100, 100, 100, 100, 100, 100]))
    max_torques: np.ndarray = field(default_factory=lambda: np.array([1.50, 1.50, 1.50, 0.28, 0.28, 0.28])) # Those on the motors (not the joints)
    robot_reach: float = 0.85 # Radius of the maximum circle

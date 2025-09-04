from dataclasses import dataclass, field
import numpy as np

@dataclass
class TestIkFlow:
    verbose: bool = False
    N_samples: int = 100  # Samples per 'discretized' pose configuration
    N_disc: int = 1  # Number of discrete configurations to test (rotational sweep)
    use_ikflow: bool = True  # Set to False to test a hard-coded joint configuration
    show_pose_duration: int = 0.03  # Seconds to show each pose

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
    csv_directory: str = "hole1/turbo_ikflow"  # Directory to save CSV files
    starting_position : float = 0.0 # Center of the line
    line_length : float = 2.0 # s meters is the total length

    # Ikflow variables
    N_samples: int = 200  # Samples per 'discretized' pose configuration
    N_disc: int = 18  # Number of discrete configurations to test (rotational sweep)
    

# ! At the end, we did not use this (it takes too long)
@dataclass
class ScrewingTurboBioik2:

    # Control variables
    verbose: bool = False # Display messages
    show_pose_duration: int = 0.05  # Seconds to show each pose
    activate_gui : bool = True  # Activate the GUI for visualization
    csv_directory: str = "screwing/turbo_bioik2"  # Directory to save CSV files


@dataclass
class ScrewingCMAES:

    # Control variables
    verbose: bool = False # Display messages
    show_pose_duration: int = 0.05  # Seconds to show each pose
    activate_gui : bool = True  # Activate the GUI for visualization
    csv_directory: str = "hole1/cma_es_ikflow"  # Directory to save CSV files
    starting_position : float = 0.0 # Center of the line
    line_length : float = 2.0 # s meters is the total length

    # Ikflow variables
    N_samples: int = 200  # Samples per 'discretized' pose configuration
    N_disc: int = 18  # Number of discrete configurations to test (rotational sweep)
    
    # cma-es variables
    x0: np.ndarray = field(default_factory=lambda: 
                           np.array([0.0, 0.05, np.radians(-90.0)]))  # initial mean mu
    sigma0 : float = 2  # initial std sigma
    popsize: int = 3  # number of individuals
    n_iter: int = 10  # number of iterations

@dataclass
class ScrewingRandom:

    # Control variables
    verbose: bool = False # Display messages
    show_pose_duration: int = 0.05  # Seconds to show each pose
    activate_gui : bool = True  # Activate the GUI for visualization
    csv_directory: str = "screwing/random"  # Directory to save CSV files

    # Ikflow variables
    N_samples: int = 20  # Samples per 'discretized' pose configuration
    N_disc: int = 1  # Number of discrete configurations to test (rotational sweep)
    
    # cma-es variables
    x0: np.ndarray = field(default_factory=lambda: 
                           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.radians(180), np.radians(-100), 
                                     np.radians(80), np.radians(-90), np.radians(-90), np.radians(45)]))  # initial mean mu
    sigma0 : float = 2  # initial std sigma
    popsize: int = 40  # number of individuals
    n_iter: int = 100  # number of iterations


@dataclass
class KukaIiwa14:
    nu: int = 7 # Number of joints
    gear_ratios: np.ndarray = field(default_factory=lambda: np.array([100, 100, 100, 100, 100, 100, 100]))
    max_torques: np.ndarray = field(default_factory=lambda: np.array([3.2, 3.2, 1.76, 1.76, 1.1, 0.4, 0.4])) # Those on the motors (not the joints)
    robot_reach: float = 0.82 # Radius of the maximum circle

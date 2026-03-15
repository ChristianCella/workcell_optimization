from dataclasses import dataclass, field
import numpy as np

@dataclass
class TestIkFlow:
    verbose: bool = True
    N_samples: int = 100  # Samples per 'discretized' pose configuration
    N_disc: int = 1  # Number of discrete configurations to test (rotational sweep)
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
class UseCaseData:

    # Control variables
    verbose: bool = False # Display messages
    show_pose_duration: int = 0.05  # Seconds to show each pose
    activate_gui : bool = True  # Activate the GUI for visualization
    mode: str = "optimization" # Either "debugging" or "optimization"

    # Ikflow variables
    Ns: int = 25  # Samples per 'discretized' pose configuration
    Nd: int = 90  # Number of discrete configurations to test (rotational sweep)
    
    # TuRBO variables
    d: int = 3
    init_rand_points: int = 35
    batch_size: int = 20
    n_desired_iterations: int = 50
    n_trust_regions: int = 5
    n_training_steps: int = 50
    lb_real: np.ndarray = field(default_factory=lambda: np.array([-0.711, -0.091, np.radians(-90)])) 
    ub_real: np.ndarray = field(default_factory=lambda: np.array([-0.4, 0.371, np.radians(90)])) 

    # Follower variables
    weights_follower: list = field(default_factory=lambda: [10.0, 0.5])
    centering_weights: np.ndarray = field(default_factory=lambda: np.ones(6))

    def __post_init__(self):

        # Computed once, right after initialization
        self.center = (self.ub_real + self.lb_real) / 2.0
        self.scale  = (self.ub_real - self.lb_real) / 2.0
        self.max_evals = self.init_rand_points + self.n_desired_iterations * self.batch_size
        self.csv_directory = self.mode

@dataclass
class Ur5eRobot:
    nu: int = 6 # Number of joints
    home_configuration: np.ndarray = field(default_factory=lambda: np.radians([0, -90, -90, -90, 90, 0]))
    gear_ratios: np.ndarray = field(default_factory=lambda: np.array([100, 100, 100, 100, 100, 100]))
    max_torques: np.ndarray = field(default_factory=lambda: np.array([1.50, 1.50, 1.50, 0.28, 0.28, 0.28])) #! Get these from experiments
    robot_reach: float = 0.85 # Radius of the maximum circle
    lb: list = field(default_factory=lambda: -2 * np.pi * np.ones(6))
    ub: list = field(default_factory=lambda: 2 * np.pi * np.ones(6))

@dataclass 
class Tools:
    fixed_radius: float = 0.0455  



from dataclasses import dataclass, field
import numpy as np

@dataclass
class TestIkFlow:
    mode: str = "full"  
    verbose: bool = True
    N_samples: int = 30  # Samples per 'discretized' pose configuration
    N_disc: int = 10  # Number of discrete configurations to test (rotational sweep)
    show_pose_duration: int = 1  # Seconds to show each pose
    use_database: bool = False # Use txt database
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
    activate_gui : bool = True  # Activate the GUI for visualization
    mode: str = "optimization" # Either "debugging" or "optimization"
    theta: int = 1 # Switch variable; theta = 1 => one overall optimization; theta = 0 => clusterized optimization

    # Ikflow variables
    Ns: int = 25  # Samples per 'discretized' pose configuration
    Nd: int = 90  # Number of discrete configurations to test (rotational sweep)
    
    # TuRBO variables
    d: int = 2
    init_rand_points: int = 1
    batch_size: int = 3
    n_desired_iterations: int = 2
    n_trust_regions: int = 3
    n_training_steps: int = 50
    lb_real: np.ndarray = field(default_factory=lambda: np.array([-0.3, -0.3]))
    ub_real: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.3]))

    # Leader variables
    weights_leader: list = field(default_factory=lambda: [10.0, 0.5])
    weights_rrt: np.ndarray = field(default_factory=lambda: np.ones(6, dtype=float))

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
    max_torques: np.ndarray = field(default_factory=lambda: np.array([1.50, 1.50, 1.50, 0.28, 0.28, 0.28])) # Those on the motors (not the joints)
    robot_reach: float = 0.85 # Radius of the maximum circle
    lb: list = field(default_factory=lambda: -2 * np.pi * np.ones(6))
    ub: list = field(default_factory=lambda: 2 * np.pi * np.ones(6))

@dataclass 
class Tools:
    hande_offset: float = 0.157  # Length of the gripper hande
    extension_offset: float = 0.2  # Length of the extension tool
    detachment_pose: list = field(default_factory=lambda: [2.0, 2.0, 2.0, 180.0, 0.0, 0.0])  # Pose to detach the extension tool
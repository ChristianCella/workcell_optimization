from dataclasses import dataclass, field
import numpy as np

@dataclass
class TestIkFlow:
    verbose: bool = True
    N_samples: int = 30  # Samples per 'discretized' pose configuration
    N_disc: int = 360  # Number of discrete configurations to test (rotational sweep)
    use_ikflow: bool = True  # Set to False to test a hard-coded joint configuration
    show_pose_duration: int = 0.01  # Seconds to show each pose

@dataclass
class VisualRedundancy:
    verbose: bool = True
    N_samples: int = 50  # Samples per 'discretized' pose configuration
    N_disc: int = 8  # Number of discrete configurations to test (rotational sweep)
    use_ikflow: bool = True  # Set to False to test a hard-coded joint configuration
    show_pose_duration: int = 0.1  # Seconds to show each pose


@dataclass
class TxtUseCase:
    verbose: bool = False
    N_samples: int = 20  # Samples per 'discretized' pose configuration
    N_disc: int = 90  # Number of discrete configurations to test (rotational sweep)
    show_pose_duration: int = 0.05  # Seconds to show each pose
    activate_gui : bool = True  # Whether to activate the GUI for visualization
    x0: np.ndarray = field(default_factory=lambda: 
                           np.array([0.0, 0.0, np.radians(100), np.radians(-95), 
                                     np.radians(100), np.radians(-95), np.radians(-95), np.radians(180)]))  # initial mean mu
    sigma0 : float = 0.5  # initial std sigma
    popsize: int = 4  # number of individuals
    n_iter: int = 3  # number of iterations

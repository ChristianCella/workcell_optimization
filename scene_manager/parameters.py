from dataclasses import dataclass, field

@dataclass
class TestIkFlow:
    verbose: bool = True
    N_samples: int = 5  # Samples per 'discretized' pose configuration
    N_disc: int = 45  # Number of discrete configurations to test (rotational sweep)
    use_ikflow: bool = False  # Set to False to test a hard-coded joint configuration
    show_pose_duration: int = 0.01  # Seconds to show each pose


@dataclass
class TxtUseCase:
    verbose: bool = False
    N_samples: int = 50  # Samples per 'discretized' pose configuration
    N_disc: int = 90  # Number of discrete configurations to test (rotational sweep)
    show_pose_duration: int = 0.1  # Seconds to show each pose
    activate_gui : bool = True  # Whether to activate the GUI for visualization
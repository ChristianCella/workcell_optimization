from dataclasses import dataclass, field

@dataclass
class TestIkFlow:
    verbose: bool = False
    N_samples: int = 130  # Samples per 'discretized' pose configuration
    N_disc: int = 180  # Number of discrete configurations to test (rotational sweep)
    use_ikflow: bool = True  # Set to False to test a hard-coded joint configuration
    show_pose_duration: int = 0.05  # Seconds to show each pose
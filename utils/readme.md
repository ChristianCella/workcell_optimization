# Utilities 🛠️
This folder contains functions and methods that are leveraged by most of the codes.

- [colors.py](https://github.com/ChristianCella/workcell_optimization/blob/txt_arto/utils/colors.py) displays a GUI that allows to select the desired color and retrieve its RGB(A) equivalent. It becomes very useful to color the frames to be used in the `.xml` files inside [ur5e_utils_mujoco](https://github.com/ChristianCella/ur5e_utils_mujoco/tree/txt_arto);
- [constant_parameters.py](https://github.com/ChristianCella/workcell_optimization/blob/txt_arto/utils/constant_parameters.py) contains teh classes where the most important parameters are inizialized. In particular:
    - $\texttt{TestIkFlow}$ is used inside [inverse_kinematics.py](https://github.com/ChristianCella/workcell_optimization/blob/txt_arto/tests/inverse_kinematics.py);
    - $\texttt{OptimizationParameters}$ is used inside [optimize_workcell.py](https://github.com/ChristianCella/workcell_optimization/blob/txt_arto/use_case/optimize_workcell.py);
- [fonts.py](https://github.com/ChristianCella/workcell_optimization/blob/txt_arto/utils/fonts.py) allows to modify the colors of the messages in the terminal;
- [ikflow_inference.py](https://github.com/ChristianCella/workcell_optimization/blob/txt_arto/utils/ikflow_inference.py) contains functions, methods and wrappers that allow to leverage the trained normalized flow inside [ikflow](https://github.com/ChristianCella/ikflow/tree/txt_arto);
- [mujoco_utils.py](https://github.com/ChristianCella/workcell_optimization/blob/txt_arto/utils/mujoco_utils.py) contains the definition of the methods that leverage the APIs of the simulator to set/get the needed values and update the resources. In addition, it contains the function `scene_manager`, that takes the path of the `.xml` files and creates a single one, called `temp_scene.xml`. The most important parameter is `case_id`:
    - `case_id="robot"`: only the robot is imported in the scene;
    - `case_id="targets"`: also `number_targets` reference frames are added to the scene (all placed coincident to the world);
    - `case_id="full"`: the complete scene contains robot, targets and the extension tool (this is placed in the world frame);
- [test_cuda_version.py](https://github.com/ChristianCella/workcell_optimization/blob/txt_arto/utils/test_cuda_version.py) displays some information about the cuda version. Useful to test if the correct version is installed.
- [transformations.py](https://github.com/ChristianCella/workcell_optimization/blob/txt_arto/utils/transformations.py) contains the functions to switch between the three most important representations for rotations (quaternions, euler angles and rotation matrices). It also contains the function `get_world_wrench`, that implements the equation $\mathbf{W^{world}}=\mathbf{R^{world}_{local}}\cdot \mathbf{W^{local}}$ (with $\mathbf{R^{world}_{local}} \in \mathbb{R}^{6\times6}$);
- [visualize_single_xml.py](https://github.com/ChristianCella/workcell_optimization/blob/txt_arto/utils/visualize_single_xml.py) allows to display a single xml.


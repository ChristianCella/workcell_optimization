# Layout optimization - ARTO project 🤖🪛➡️
This repo allows to optimize a robotic ell, regardless of the robot, the position of the workpiece and the number of screws. An extended version to accoutn for general contact applications is under development.

---

### **Requirements and OS** 🪟 <a name="Requirements"></a>
The work is meant to be python-based, without the necessity to be enclosed in a ROS/ROS2 framework. The suggested requirements are:
- ❇️ Windows 11 / Ubuntu 22.04
- ❇️ Python 3.10.0 (higher versions may be good too; not tested though)
- ❇️ VSCode or PyCharm (tested on VSCode)

---

### **Project structure** 🏗️ <a name="Structure"></a> 
This project relies on utilities for mujoco ([link](https://github.com/ChristianCella/ur5e_utils_mujoco.git)) and an IK solver ([link](https://github.com/ChristianCella/ikflow.git)), as depicted in the following:
```
├── ur5e_utils_mujoco/           # utilities for mujoco and ikflow 
    ├── ...
    ├── ur5e.xml
    ├── ur5e.urdf
├── workcell_optimization/       # this package (explained in the following)
└── ikflow/                      # normalizing flow for inverse kinematics   
```

For what concerns [workcell_optimization](https://github.com/ChristianCella/workcell_optimization.git), the structure is currently the following:

```
├── bayesian_optimizers/         # codes for Bayesian Optimization
├── tests/                       # test codes
├── use_case/                    # real tests
├── utils/                       # some utilities codes
├── .gitignore                             
├── readme.md
└── requirements.txt   
```

---

### **Installation procedure** ▶️ <a name="Install"></a> 

Create a folder (for example 'robotic_conatct_operations') and clone the required packages (```git clone ...```):

```
https://github.com/ChristianCella/workcell_optimization.git
https://github.com/ChristianCella/ur5e_utils_mujoco.git
https://github.com/ChristianCella/ikflow.git
```

open a terminal in the directory ```../workcell_optimization``` and create a virtual environment called ```.venv``` (if using Windows 11):

```
py 3.10 -m venv .venv
.venv\Scripts\activate
```
At this point, install all dependencies with ```pip install -r requirements.txt```. Now, go to the directory of ikflow (from ```../workcell_optimization```, just do ```cd..``` and then ```cd ikflow```) and follow these steps:

```
py -3.10 -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pytorch-lightning wandb jrl-robots numpy tqdm
pip install -e .
```

Remember to work on a branch, not on main! To do so, create a branch:

```
git branch "desired_branch_name"
```

Then, checkout on that branch:

```
git checkout "desired_branch_name"
```

Before proceeding, always make sure that you really checked out on that branch with ```git branch -a```.

---

### **Tests** 🔎 <a name="Tests"></a> 
There are currently two folders inside [tests](https://github.com/ChristianCella/Screwdriving_MuJoCo/tree/main/tests):
- [Physics](https://github.com/ChristianCella/Screwdriving_MuJoCo/tree/main/tests/Physics): contains all the python files to solve the statics/dynamics of the robot.
    - [impose_q.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Physics/impose_q.py): this code allows to retrieve one of the joint configurations of the robot, based on a target Cartesian pose. This code is NOT meant for simulations involving physics: the robot is instantaneosuly configured according to the solution of the inverse kinematics, but no control is applied at the joints;
    - [test_PID](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Physics/test_PID.py): this code is meant to provide an alternative to [test_wrench](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/test_wrench.py), where the idea is to not compute the torques, but to read them at the joints. For this purpose, a PID is applied, with the aim of keeping a desired joint configuration. Howeveer, despite the codes does not work bad, it is not as good as [test_wrench.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Physics/test_wrench.py);
    - [test_wrench.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Physics/test_wrench.py): the logic in this code is similar to what we are going to use. Some joint configurations are specified and the inverse dynamics (i.e. torques at the joints) is computed: the wrench is specified in terms of world coordinates, therefore also the jacobian must be from world to the point (site) where the wrench is applied. A PD controller is used to compensate for small numerical errors. 
- [Redundancy](https://github.com/ChristianCella/Screwdriving_MuJoCo/tree/main/tests/Redundancy): Contains all the python files to solve the redundancy in three different ways.
    - [global_redundancy.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Redundancy/global_redundancy.py): allows to solve the problem $q^* = \argmin_{q \in [\text{joint limits}]} J(q)$ in a global sense, by solving two optimization problems sequentially: first, it uses a custom implementation of the [BioIK](https://github.com/TAMS-Group/bio_ik) solver to efficiently sample the domain; subsequently, a local gradient-descent algorithm is used to fine-tune the solution.
    - [nl_constrained_redundancy.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Redundancy/nl_constrained_redundancy.py): allows to optimize the robot joint configuration by solving the following non-linear constarined optimization problem $\min_{q} U(q)$ s.t. $f(q)=x$. Unlike the example provided in [test_redundancy.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Redundancy/test_redundancy.py), in this case the library ```scipy.optimize.minimize``` is employed. Despite the formalism and the elegant nature of the problem beneath, achieving the optimal result is time-consuming and usually takes more than the procedure implemented in [test_redundancy.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Redundancy/test_redundancy.py);
    - [test_redundancy.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Redundancy/test_redundancy.py): this code allows to find the joint configuration that optimizes a certain objective function $U(q)$. The selected resolutin technique is the $\textit{Null-space Gradient}$ method, where the optimal configuration $q^*$ is iteratively found according to $q_{k+1} \gets q_k - \alpha N (\frac{\partial U(q)}{\partial q})^T$. In the test, the functional $U(q) = \sqrt{\det{(J^T J)}}$ and the solution is projected back on the feasible manifold at the very end, to enforce the constraint $x = f(q)$ (basically, the code solves $q_{final} = \argmin_q ||q - q_{opt}||^2$ s.t. $x^*=f(q)$):.

---

### **bayesian optimizers** 📈 <a name="Bayesian"></a> 
This folder contains the entry-level files for Bayesian Optimization.
- [docs](https://github.com/ChristianCella/Screwdriving_MuJoCo/tree/main/bayesian_optimizers/docs): contains a pdf for some theory and a very simple code to understand the basics;
- [custom](https://github.com/ChristianCella/Screwdriving_MuJoCo/tree/main/bayesian_optimizers/custom): some hand-made codes for more structured examples;
- [botorch_based](https://github.com/ChristianCella/Screwdriving_MuJoCo/tree/main/bayesian_optimizers/botorch_based): these are some of the most advanced codes; more specifically, the implementations of TuRBO and SCBO, that you cna find at [this](https://botorch.org/) link on BoTorch. On this site there is also a lot of useful staff: take a look at it.

---

### **Troubleshooting** 🛠️ <a name="Troubleshooting"></a> 
1. **General installation problems:**
    If you experience any problem when installing packages, maybe it is beacuse you do not have the 'LongPathEnabled' option in your registry. To fix it, open your system registers (regedit) and navigate to
    ```
    Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
    ```
    At this point, go to the cell 'LongPathEnabled' and change the value from 0 to 1.
2. **Numpy, torch and torchvision:**
To check the CUDA version installed you can run the command
    ```
    nvidia-smi
    ```
    If you have no CUDA installed, you can access the following website and install the Toolkit:
    - https://developer.nvidia.com/cuda-downloads

    CUDA needs a specific version of torch and torchvision. Look at the following webiste to install the correct one:
    - https://pytorch.org/get-started/locally/

    In my case, since I dispose of CUDA 12.4, I have to run the following command:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

---

### **Contacts** 📧 <a name="Contacts"></a> 
<img align="center" height="40" src="https://avatars.githubusercontent.com/u/113984059?v=4"> Christian Cella: christian.cella@polimi.it

<img align="center" height="40" src="https://avatars.githubusercontent.com/u/127955558?v=4"> Alessandro Casciani: alessandro.casciani@mail.polimi.it


# Layout optimization - ARTO project 🤖🕹️✈️
This work allows to optimize a robotic workcell, regardless of the robot, the position of the workpiece and the number of target locations. This branch contains only the codes for the ARTO project, developed with TXT e-solutions.

---

### **Requirements and OS** 🐧💻 <a name="Requirements"></a>
The work is meant to be python-based, without the necessity to be enclosed in a ROS/ROS2 framework. The suggested requirements are:
- Ubuntu 22.04 (tested also on Ubuntu 20.04 and Windows 11)
- Python 3.10.0 (newer versions have not been tested yet)

---

### **Project structure** 🗂️ <a name="Structure"></a> 
This project relies on utilities for mujoco ([link](https://github.com/ChristianCella/ur5e_utils_mujoco.git)) and an IK solver ([link](https://github.com/ChristianCella/ikflow.git)), as depicted in the following:

```
├── ikflow/       # Normalizing flow for Inverse Kinematics
├── TuRBO/       # Package for Bayesian Optimization
├── ur5e_utils_mujoco/           # utilities for mujoco and ikflow 
    ├── ...
    ├── simple_obstacles/
    ├── ur5e/
        ├── ...
        ├── ur5e.xml
        ├── ur5e.urdf
└── workcell_optimization/       # Shared package to optimize robotic applications  
```

---

### **Installation procedure** ▶️ <a name="Install"></a> 

Create a folder (for example 'robotic_conatct_operations') and place in it all the required packages:

```
git clone https://github.com/ChristianCella/workcell_optimization.git
git clone https://github.com/ChristianCella/ikflow.git
git clone https://github.com/ChristianCella/ur5e_utils_mujoco.git
git clone https://github.com/ChristianCella/TuRBO.git
```
- #### **Workcell optimization package** 📈 <a name="workcell_optimization_package"></a> 
    open a terminal in the directory ```../workcell_optimization``` and create a virtual environment called ```.venv```:

    ```
    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    NOTE: Be sure to have all the required packages (CUDA in particular. If some problems arise, look at the Troubleshooting section).

- #### **Ikflow** 🦿 <a name="ikflow"></a> 

    Now, go to the directory of ikflow (from ```../workcell_optimization```, just do ```cd..``` and then ```cd ikflow```) and follow these steps:

    ```
    python3.10 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
    pip install pytorch-lightning wandb jrl-robots numpy tqdm
    pip install -e .
    ```
    NOTE: switch to the branch ```txt_arto``` for this package. Inside ```ikflow```, another folder with the same name is present. Inside this one, manually create a folder called ```weights```, and copy in it the learned weights for the ur5e robot.

- #### **Mujoco utilities for the ur5e robot** ⚙️ <a name="ur5e_mujoco"></a> 
    No specific recommmendation for this package. This is a custom version of the original repository [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie), that only focuses on the ur5e robot. Despite not mandatory, switch on the branch ```txt_arto``` also for this package.

- #### **Bayesian optimization with TuRBO** 🧠 <a name="turbo_bayesian"></a> 
    No specific recommmendation for this package. The repository linked above is a fork of [this](https://github.com/uber-research/TuRBO) package. Despite not mandatory, switch on the branch ```txt_arto``` also for this package.
---

### **Troubleshooting** 🛠️ <a name="Troubleshooting"></a> 
Most of the CUDA packages have been removed from the ```requirements.txt``` file, since they depend on the available hardware. Follow the instructions below to install the necessary packages.

- **Numpy, torch and torchvision:**
To check the CUDA version installed you can run the command
    ```
    nvidia-smi
    ```
    If you have no CUDA installed, you can access the following website and install the Toolkit:
    - https://developer.nvidia.com/cuda-downloads

    CUDA needs a specific version of torch and torchvision. Look at the following webiste to install the correct one:
    - https://pytorch.org/get-started/locally/

    As an example, if you dispose of CUDA 12.4, use the following command:
    ```bash
    pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
    ```
    To check if all the packages are correctly installed, run the code ```test_cuda_version.py``` inside ```utils```. The output should be similar to:
    ```
    PyTorch version: 2.6.0+cu124
    Torchvision version: 0.21.0+cu124
    CUDA availability: True
    The device for the training is: NVIDIA GeForce RTX 4070 Laptop GPU
    ```

---

### **Contacts** 📧 <a name="Contacts"></a> 
<img align="center" height="40" src="https://avatars.githubusercontent.com/u/113984059?v=4"> Christian Cella: christian.cella@polimi.it



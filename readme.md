# Robotic screwdriving application 🪛
This repo allows to optimize a robotic screwdriving cell, regardless of the robot, the position of the workpiece and the number of screws. An extended version to accoutn for general contact applications is under development.

---

### **Requirements and OS** 🪟 <a name="Requirements"></a>
The work is meant to be python-based, without the necessity to be enclosed in a ROS framework. The suggested requirements are:
- ❇️ Windows 11 / Ubuntu 22.04
- ❇️ Python 3.10.0 (higher versions may be good too; not tested though)
- ❇️ VSCode or PyCharm (tested on VSCode)

---

### **Installation procedure** <a name="Install"></a> ▶️

Install the repository:

```
git clone https://github.com/ChristianCella/Screwdriving_MuJoCo.git
```

open a terminal in the directory ```../Screwdriving_MuJoCo``` and create a virtual environment called ```.venv``` (if using Windows 11):

```
python -m venv .venv
```
and activate it from the terminal typing ```.venv\Scripts\activate``` (for Windows; if you need to deactivate it for some reason, type ```deactivate```). At this point, install all dependencies with ```pip install -r requirements.txt```. Remember to work on a branch, not on main! To do so, create a branch:

```
git branch "desired_branch_name"
```

Then, checkout on that branch:

```
git checkout "desired_branch_name"
```

Before proceeding, always make sure that you really checked out on that branch with ```git branch -a```.

---

### **Project structure** <a name="Structure"></a> 🏗️
At the moment, the structure is the following:

```
├── tests/                       # test codes
├── universal_robots_ur5e/       # xml code to setup the MuJoCo scene
├── .gitignore                             
├── readme.md
└── requirements.txt   
```

---

### **Tests** <a name="Tests"></a> 🔎
There are currently two folders inside [tests](https://github.com/ChristianCella/Screwdriving_MuJoCo/tree/main/tests):
- [Physics](https://github.com/ChristianCella/Screwdriving_MuJoCo/tree/main/tests/Physics): contains all the python files to solve the statics/dynamics of the robot.
    - [impose_q.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Physics/impose_q.py): this code allows to retrieve one of the joint configurations of the robot, based on a target Cartesian pose. This code is NOT meant for simulations involving physics: the robot is instantaneosuly configured according to the solution of the inverse kinematics, but no control is applied at the joints;
    - [test_PID](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Physics/test_PID.py): this code is meant to provide an alternative to [test_wrench](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/test_wrench.py), where the idea is to not compute the torques, but to read them at the joints. For this purpose, a PID is applied, with the aim of keeping a desired joint configuration. Howeveer, despite the codes does not work bad, it is not as good as [test_wrench.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Physics/test_wrench.py);
    - [test_wrench.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Physics/test_wrench.py): the logic in this code is similar to what we are going to use. Some joint configurations are specified and the inverse dynamics (i.e. torques at the joints) is computed: the wrench is specified in terms of world coordinates, therefore also the jacobian must be from world to the point (site) where the wrench is applied. A PD controller is used to compensate for small numerical errors. 
- [Redundancy](https://github.com/ChristianCella/Screwdriving_MuJoCo/tree/main/tests/Redundancy): Contains all the python files to solve the redundancy in three different ways.
    - [global_redundancy.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Redundancy/global_redundancy.py): allows to solve the problem $q^* = \argmin_{q \in [\text{joint limits}]} J(q)$ in a global sense, by solving two optimization problems sequentially: first, it uses a custom implementation of the [BioIK](https://github.com/TAMS-Group/bio_ik) solver to efficiently sample the domain; subsequently, a local gradient-descent algorithm is used to fine-tune the solution.
    - [nl_constrained_redundancy.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Redundancy/nl_constrained_redundancy.py): allows to optimize the robot joint configuration by solving the following non-linear constarined optimization problem $\min_{q} U(q)$ s.t. $f(q)=x$. Unlike the example provided in [test_redundancy.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Redundancy/test_redundancy.py), in this case the library ```scipy.optimize.minimize``` is employed. Despite the formalism and the elegant nature of the problem beneath, achieving the optimal result is time-consuming and usually takes more than the procedure implemented in [test_redundancy.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Redundancy/test_redundancy.py);
    - [test_redundancy.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/Redundancy/test_redundancy.py): this code allows to find the joint configuration that optimizes a certain objective function $U(q)$. The selected resolutin technique is the $\textit{Null-space Gradient}$ method, where the optimal configuration $q^*$ is iteratively found according to $q_{k+1} \gets q_k - \alpha N (\frac{\partial U(q)}{\partial q})^T$. In the test, the functional $U(q) = \sqrt{\det{(J^T J)}}$ and the solution is projected back on the feasible manifold at the very end, to enforce the constraint $x = f(q)$ (basically, the code solves $q_{final} = \argmin_q ||q - q_{opt}||^2$ s.t. $x^*=f(q)$):.


## Yet to be done

Understand how to introduce motion planning and trajectory tracking. Usually, industrial robots are featured with a controller that automatically compensates gravity, but in this case we have to account for it and also for an external wrench. If this is true, motion planning would not be particularly difficult.

---

### **Contacts** <a name="Contacts"></a> 📧
<img align="center" height="40" src="https://avatars.githubusercontent.com/u/113984059?v=4"> Christian Cella: christian.cella@polimi.it

<img align="center" height="40" src="https://avatars.githubusercontent.com/u/83009256?v=4"> Alessandro Casciani: alessandro.casciani@mail.polimi.it


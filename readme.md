# Robotic screwdriving application

## Installation
Requirements:
- Python 3.10.0 (higher versions may be good too; not tested though)
- VSCode

After cloning the repo with this command:

```
git clone https://github.com/ChristianCella/Screwdriving_MuJoCo.git
```

open a terminal in the directory ```../Screwdriving_MuJoCo``` and create a virtual environment called ```.venv```:

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

Check that you rally checked out on that branch with ```git branch -a```.

## Strcucture of the project
At the moment, the structure is the following:

```
├── tests/                       # test codes
├── universal_robots_ur5e/       # xml code to setup the MuJoCo scene
├── .gitignore                             
├── readme.md
└── requirements.txt   
```

## Tests
- [impose_q.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/impose_q.py): this code allows to retrieve one of the joint configurations of the robot, based on a target Cartesian pose. This code is NOT meant for simulations involving physics: the robot is instantaneosuly configured according to the solution of the inverse kinematics, but no control is applied at the joints. (TODO: is it possible to fix the joints in some ways and read the torques?);
- [test_PID](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/test_PID.py): this code is meant to provide an alternative to [test_wrench](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/test_wrench.py), where the idea is to not compute the torques, but to read them at the joints. For this purpose, a PID is applied, with the aim of keeping a desired joint configuration. Howeveer, despite the codes does not work bad, it is not as good as [test_wrench.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/test_wrench.py);
- [test_wrench.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/tests/test_wrench.py): the logic in this code is similar to what we are going to use. Some joint configurations are specified and the inverse dynamics (i.e. torques at the joints) is computed: the wrench is specified in terms of world coordinates, therefore also the jacobian must be from world to the point (site) where the wrench is applied. A PD controller is used to compensate for small numerical errors. 

## Yet to be done

Understand how to introduce motion planning and trajectory tracking. Usually, industrial robots are featured with a controller that automatically compensates gravity, but in this case we have to account for it and also for an external wrench. If this is true, motion planning would not be particularly difficult.
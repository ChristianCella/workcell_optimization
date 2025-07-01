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
and activate it from the terminal typing ```.venv\Scripts\activate``` (for Windows; if you need to deactivate it for some reason, type ```deactivate```). At this point, install all dependencies with ```pip install -r requirements.txt```. Remember to work on a branch, not on main!

## Strcucture of the project
Yet to be decided

## Utilities
- [forward_kin.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/forward_kin.py): retrieves the Cartesian pose of a desired site, after applying in open loop the torques needed to cancel out gravity for a specified joint configuration of the robot;
- [impose_q.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/impose_q.py): leverages the github repository [mink](https://github.com/kevinzakka/mink/tree/main), that relies on [pinocchio](https://github.com/stack-of-tasks/pinocchio) under the hood, to retrieve one of the joint configurations of the robot, based on a target Cartesian pose. This code is NOT meant for simulations involving physics: the robot is instantaneosuly configured according to the solution of the inverse kinematics, but no control is applied at the joints. (TODO: is it possible to fix the joints in some ways and read the torques?);
- [open_loop_torques.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/open_loop_torques.py): This code will soon be removed. The idea was to have 2 control modes: the first one (move_robot = True) allowing to reach the target joint configuration computing in open-loop the torques at runtimne, while using a PID controller to correct small errors (NOTE: the way the wrench was trated was not correct!); the second (move_robot = True) was meant to instantly place the robot in a specific point in space, and forcing the intial configuration to be the desired one. At this point, the torques are computed to stay in that configuration, while a PD allows to correct small numewrical mistakes; 
- [test_PID](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/test_PID.py): this code was meant to provide an alternative to [open_loop_torques.py](), where the idea was to not compute the trques, but to read them at the joints. For this purpose, a PID was applied, with the aim of keeping adesired joint configuration. Howeveer, despite the codes does not work bad, it is not as good as [test_wrench.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/test_wrench.py), that is the improved version of [open_loop_torques.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/open_loop_torques.py);
- [test_wrench.py](https://github.com/ChristianCella/Screwdriving_MuJoCo/blob/main/test_wrench.py): the logic in this code is simialr to what we are going to use. Some joint configurations are specified and the inverse dynamics is computed: the wrench is specified in terms of world coordinates, therefore also the jacobian must be from world to the point (site) where the wrench is applied. Once again, a PD controller is udes to compensate for small numerical errors. 

## Yet to be done

Understand how to introduce motion planning and trajectory tracking. Usually, industrial robots are featured with a controller that automatically compensates gravity, but in this case we have to account for it and also for an external wrench. If this is true, motion planning would not be particularly difficult.
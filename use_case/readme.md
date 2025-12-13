# Use case 🧩
The goal of the files in this folder is to implement the layout optimization process. The algorithm determines the optimal coordinates $x_b$ and $y_b$ of the robot base (in the world coordinates) such that the manipulator can exert the prescribed wrench at each target location (both queried from a database), at the expense of the smallest possible set of joint torques (main goal of the optimization).

The two problems ($\textit{i.e.,}$ torque minimization and optimal joints configuration) can be arranged according to the leader-follower paradigm:
- leader problem - scalarization of mean torques and maximimzation of the number of reachable targets:

    $f_{\tau}=\frac{1}{N}\sum_{j=1}^N \lVert \boldsymbol{\hat{\tau}_j}\rVert_2 \quad f_{\text{reach}}=1-\frac{n}{N} \quad f_{\text{leader}}=w_{\tau}f_{\tau}+w_{\text{reach}}f_{\text{reach}}$

    where $n$ is the number of targets that can be reached by the robot, while $N$ is the total number of targets in the cluster.

- follower(s) problem(s) - The number of followers depends on the number of targets in each cluster. The objective function for each follower is the scalarization of the inverse manipulability and the centering within the joint limits:

    $f_{\delta}=\frac{1}{\sqrt{\det{\mathbf{J}\mathbf{J}^{\top}}}} \quad f_{\text{joints}}=(\mathbf{q}-\mathbf{m})^{\top}\mathbf{T}(\mathbf{q}-\mathbf{m}) \quad f_{\text{follower}}=w_{\delta}f_{\delta}+w_{\text{joints}}f_{\text{joints}}$

    where $\mathbf{J}$ is the robot Jacobian, $\mathbf{m} = (\mathbf{q_L} + \mathbf{q_U})/2$ is the vector of midpoints, while $\mathbf{T}=\mathbf{C}^{-\top}\mathbf{C}^{-1}$ (notice that $\mathbf{C}=diag[(\mathbf{q_L} - \mathbf{q_U})/2]$ is the matrix of half-ranges). Actually, some weights can be defined to prioritize some joints, but for now ```centering_weights``` contains unitary weights.

The leader is solevd through $\texttt{TuRBO}$ (look at [this](https://github.com/ChristianCella/TuRBO/tree/txt_arto) repository), while the followers are solved by means of $\texttt{ikflow}$ (look at [this](https://github.com/ChristianCella/ikflow/tree/txt_arto) repository).
The code ```optimize_workcell.py``` implements the complete framework, while ```visualize_optimal_results.py``` imports the needed data to visualize the optimal layout obtained. Some constant parameters shared by both the files can be set inside ```.../utils/constant_parameters.py``` in the class ```OptimizationParameters```.

---

## Frames 🔴🟢🔵
In the code, the variables A_i_j represent the homogneous matrices $\mathbf{A^i_j}$, expressing the roto-translation between the starting frame $\mathbf{\{h_i\}}$ and $\mathbf{\{h_j\}}$. The names are referred to the following scheme.

---

## Variables 🔢
The optimization is performed as a function of a set of parameters.

- ```theta``` defines the optimization type (switch appearing in the first sheme of the paper). If $\theta=1$, one single optimal layout is determined for all targets; if $\theta=0$, a specific layout is associated to each cluster individually. How the clusterization happens should be defined in the database.
- ```mode```: allows to perform either the real optimization or a 'fake' procedure (mode = "debugging") to test rapidly some changes in the code.
- ```weights_leader```, ```weights_follower``` are the weights to be used in the leader and the follower(s).

In addition, despite the code is structured as visible in Algs. 1-2 of the associated paper, the implementation is based on the following functions: 

- ```make_simulator``` sets everything needed to evaluate one layout;
- ```run_sim```, defined inisde the previous, executes one simulation and returns the fitness value. This function can be called at a later stage with different layout parameters.
- ```objective_single``` is the black-box objective function of the leader problem leveraged by TuRBO to evaluate the process. It runs ```run_sim```, retrieves the primary and secondary metrics of the leader for each individual of the batch, for each batch until the number of iterations is over. 
- ```decode``` allows to restore the phyisical meaning of the optimization variables (TuRBO works on an a-dimensional domain $\in [-1;1]$)

Finally, unlike most optimization schemes, TuRBO must be fed with the maximum number of evaluations (```max_evals```), rather than with the desired number of iterations ```n_desired_iterations```. The most important parameters are:

- ```layout``` is the variable where $\boldsymbol{\xi}$ is contained. The optimization vector can contain any quantity, but in this initial we set it as $\boldsymbol{\xi}=[x_b,y_b]$
- ```d``` is the dimension of the optimization vector
- ```init_rand_points``` is the number of initial random evaluations before TuRBO starts modeling.
- ```batch_size``` is the number of points evaluated in parallel per iteration.
- ```n_desired_iterations``` is the number of TuRBO optimization iterations to run.
- ```n_trust_regions``` is the number of independent trust regions maintained by TuRBO.
- ```n_training_steps``` is the number of training steps for the surrogate model ($\textit{e.g.,}$ a Gaussian Process) per iteration.

In the end, starting from the desired number of iterations, the total number of required evaluations is computed automatically as ```self.max_evals = self.init_rand_points + self.n_desired_iterations * self.batch_size```

For what concerns $\texttt{ikflow}$, the most important parameters are 

- ```Ns``` is the number of retrieved joints configurations for the same Cartesian pose.
- ```Nd``` allows to explit task redundancy. In case of the $\texttt{Robotiq hande}$ gripper alone, Nd=2 allows to consider the prescribed Cartesian pose or the one rotated of $\pi$ around z. For the extension tool, Nd = 90 allows to retrieve the joints configurations for incremental rotations around the z axis (each new Cartesian pose has the same $\{x,y,z,\theta_x, \theta_y\}$), but a $\theta_z = \theta_z + i \cdot 2\pi/Nd$
---

## Visualize best solution 🔍
This code is meant to display the optimal results. In case of a single tradeoff solution ($\theta = 1$), it is sufficient to import the files:

- ```best_layout_cluster_1.csv```
- ```best_joints_configs_cluster_1.csv```

In case of clustered optimization ($\theta = 0$) you can decide the cluster of targets to be displayed by means of the variable ```cluster_to_visualize```. In this situation, the correct .csv files will be opened automatically.
# Use case procedure
The simulation results are compared to the solutions proposed by 10 participants. The goal of each test is simple: the M8 screw needs to be inserted in the threaded hole. To obtain meaningful results, each test has been performed following the procedure reported below:

### Rules
- Participants do not know the result of the optimization, neither the fact that their proposal will be compared to the results of the simulations.
- Before each test, the robot posture is q = [90, -90, 0, -90, -90, 0].
- Each participant is asked to place the piece anywhere on the table, choose an appropriate support connecting the screwdriver to the flange, and choose the robot configuration they desire.
- The correct payload is set for the robot, as a function of the selected support
- The program read_effort.py is executed, and the trend of the torques is obtained.
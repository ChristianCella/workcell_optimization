clear; clc; close all
manipulator = loadrobot("universalUR5e", DataFormat="row", Gravity=[0 0 -9.81]);
%q = [deg2rad(180) deg2rad(-100) deg2rad(80) deg2rad(-90) deg2rad(-90) deg2rad(-45)];
q = [0.3, -1.308, 1.214, 1.663, -4.713, -3.163];

% Baseline (gravity only)
tau_g = inverseDynamics(manipulator, q);

% Apply a noticeable wrench at tool0 (N and N·m, expressed in tool0 frame):
% Try a pure force along tool Z and a wrist torque about Z
wrench = [0 0 -30 0 0 -30];  % Reaction forces
fext   = externalForce(manipulator, "tool0", wrench, q);
tau_fx = inverseDynamics(manipulator, q, [], [], fext);

disp([tau_fx])   % last row is the change due to the wrench

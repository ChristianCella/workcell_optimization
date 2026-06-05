from parameters import Ur5eRobot, GoFaRobot

robot_to_use = "gofa5" # "ur5e" or "gofa5"
tool_to_use = "painting_gun" # "welding_gun" or "screwdriver"
piece_to_use = "reconstructed" # "cube", "t_shape", or "reconstructed"
ik_solver_to_use = "dls" # "ikflow" or "dls"
save_data = False
import_data = False
v_red_per = 0.2
a_red_per = 0.2

if robot_to_use == "ur5e":
    rob_folder = "ur5e"
    rob_name = "UR5e.xml"
    rob_par = Ur5eRobot()
    joint_names = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]
    ik_link_training = "wrist_3_link"
elif robot_to_use == "gofa5":
    rob_folder = "gofa5"
    rob_name = "GoFa5.xml"
    rob_par = GoFaRobot()
    joint_names = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
    ik_link_training = "tool0"
else:
    raise ValueError(f"Unknown robot type: {robot_to_use}")

if tool_to_use == "welding_gun":
    tool_name = "welding_gun.xml"
elif tool_to_use == "screwdriver":
    tool_name = "screwdriver.xml"
elif tool_to_use == "painting_gun":
    tool_name = "painting_gun.xml"
else:
    raise ValueError(f"Unknown tool type: {tool_to_use}")

if piece_to_use == "cube":
    pie_name = "cube.xml"
elif piece_to_use == "t_shape":
    pie_name = "t_shape.xml"
elif piece_to_use == "reconstructed":
    pie_name = "reconstructed.xml"
else:
    raise ValueError(f"Unknown piece type: {piece_to_use}")
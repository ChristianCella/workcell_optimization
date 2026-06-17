from parameters import Ur5eRobot, GoFaRobot, FanucCrx10iaLRobot, DoosanA0509Robot, FairinoFR10Robot

robot_to_use = "fanuc_crx_10ia_l" # "ur5e" or "gofa5" or "fanuc_crx_10ia_l" or "doosan_a0509" or "fairino_FR10"
tool_to_use = "painting_gun" # "welding_gun" or "screwdriver" or "painting_gun"
piece_to_use = "svin000412" # "cube", "t_shape", "reconstructed", "svin000412"
ik_solver_to_use = "trackik" # dls or trackik (mink, clik => avoid)
verbose = False
save_data = False
import_data = False
v_red_per = 0.1
a_red_per = 0.1

if robot_to_use == "ur5e":
    rob_folder = "ur5e"
    rob_name = "UR5e.xml"
    rob_par = Ur5eRobot()
    joint_names = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]
    ik_base_link = "base_link"
    ik_link_training = "wrist_3_link"
elif robot_to_use == "gofa5":
    rob_folder = "gofa5"
    rob_name = "GoFa5.xml"
    rob_par = GoFaRobot()
    joint_names = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
    ik_base_link = "base_link"
    ik_link_training = "tool0"
elif robot_to_use == "fanuc_crx_10ia_l":
    rob_folder = "fanuc_crx_10ia_l"
    rob_name = "fanuc_crx_10ia_l.xml"
    rob_par = FanucCrx10iaLRobot()
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6"] 
    ik_base_link = "base"
    ik_link_training = "flange"
elif robot_to_use == "doosan_a0509":
    rob_folder = "doosan_a0509"
    rob_name = "doosan_a0509.xml"
    rob_par = DoosanA0509Robot()
    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    ik_base_link = "base_link"
    ik_link_training = "tool0"
elif robot_to_use == "fairino_FR10":
    rob_folder = "fairino"
    rob_name = "fairino_fr10.xml"
    rob_par = FairinoFR10Robot()
    joint_names = ["j1", "j2", "j3", "j4", "j5", "j6"]
    ik_base_link = "base_link"
    ik_link_training = "wrist3_link"
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
elif piece_to_use == "svin000412":
    pie_name = "svin000412.xml"
else:
    raise ValueError(f"Unknown piece type: {piece_to_use}")
from parameters import Ur5eRobot, GoFaRobot

robot_to_use = "ur5e" # "ur5e" or "gofa5"
tool_to_use = "welding_gun" # "welding_gun" or "screwdriver"
piece_to_use = "cube"
ik_solver_to_use = "dls" # "ikflow" or "dls"

if robot_to_use == "ur5e":
    rob_folder = "ur5e"
    rob_name = "UR5e.xml"
    rob_par = Ur5eRobot()
elif robot_to_use == "gofa5":
    rob_folder = "gofa5"
    rob_name = "GoFa5.xml"
    rob_par = GoFaRobot()
else:
    raise ValueError(f"Unknown robot type: {robot_to_use}")

if tool_to_use == "welding_gun":
    tool_name = "welding_gun.xml"
elif tool_to_use == "screwdriver":
    tool_name = "screwdriver.xml"
else:
    raise ValueError(f"Unknown tool type: {tool_to_use}")

if piece_to_use == "cube":
    pie_name = "cube.xml"
else:
    raise ValueError(f"Unknown piece type: {piece_to_use}")
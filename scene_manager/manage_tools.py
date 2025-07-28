from lxml import etree
import mujoco
import mujoco.viewer
import os
import copy

def attach_tool_to_robot(
    base_dir="",
    scene_filename="scene.xml",
    robot_filename="ur5e.xml",
    tool_filename="screwdriver.xml",
    output_filename="temp_scene_with_tool.xml",
    target_body_name="ee_frame_visual_only"
):
    scene_path = os.path.join(base_dir, "ur5e_utils_mujoco", scene_filename)
    ur5e_path = os.path.join(base_dir, "ur5e_utils_mujoco", robot_filename)
    screw_path = os.path.join(base_dir, "ur5e_utils_mujoco", tool_filename)
    output_path = os.path.join(base_dir, "ur5e_utils_mujoco", output_filename)

    # Load scene
    with open(scene_path, 'r') as f:
        scene_tree = etree.parse(f)
    scene_root = scene_tree.getroot()

    # Load robot definition and inject it in place of <include>
    with open(ur5e_path, 'r') as f:
        ur5e_tree = etree.parse(f)
    ur5e_root = ur5e_tree.getroot()

    for include in scene_root.findall(".//include"):
        if include.get("file") == robot_filename:
            parent = include.getparent()
            index = parent.index(include)
            for el in ur5e_root:
                parent.insert(index, el)
                index += 1
            parent.remove(include)

    # Load screwdriver
    with open(screw_path, 'r') as f:
        screw_tree = etree.parse(f)
    screw_root = screw_tree.getroot()

    screw_body = screw_root.find(".//body[@name='screw_top']")
    if screw_body is None:
        raise RuntimeError("Could not find body named 'screw_top' in screwdriver.xml")

    screw_body_copy = copy.deepcopy(screw_body)

    # Find insertion point and attach
    ee_frame = scene_root.find(f".//body[@name='{target_body_name}']")
    if ee_frame is None:
        raise RuntimeError(f"Target body '{target_body_name}' not found in merged XML")

    ee_frame.append(screw_body_copy)

    # Write combined XML to file
    combined_xml = etree.tostring(scene_root, pretty_print=True).decode()
    with open(output_path, "w") as f:
        f.write(combined_xml)
    
    return output_path


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_path = attach_tool_to_robot(base_dir=base_dir, tool_filename="screwdriver.xml")

    # Launch viewer
    model = mujoco.MjModel.from_xml_path(output_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()

from lxml import etree
import os
import sys
import copy
import shutil
import mujoco
import mujoco.viewer

def create_reference_frames(base_dir, vanilla_scene, n_targets):

    # Setup
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, vanilla_scene)


    # Now load and modify the XML
    with open(xml_path, 'r') as f:
        xml_string = f.read()

    # Parse and modify
    root = etree.fromstring(xml_string)

    # Add a new frame body
    worldbody = root.find(".//worldbody")
    for i in range(n_targets):
        target_name = f"reference_target_{i+1}"
        x_axis_name = f"reference_x_axis{i+1}"
        y_axis_name = f"reference_y_axis{i+1}"
        z_axis_name = f"reference_z_axis{i+1}"
        body = etree.SubElement(worldbody, "body", name=target_name, pos="0.1 0.1 0.1")
        etree.SubElement(body, "geom", name=x_axis_name, type="capsule", fromto="0 0 0 0.1 0 0", size="0.005", rgba="1 0 0 0.5", contype="0", conaffinity="0")
        etree.SubElement(body, "geom", name=y_axis_name, type="capsule", fromto="0 0 0 0 0.1 0", size="0.005", rgba="0 1 0 0.5", contype="0", conaffinity="0")
        etree.SubElement(body, "geom", name=z_axis_name, type="capsule", fromto="0 0 0 0 0 0.1", size="0.005", rgba="0 0 1 0.5", contype="0", conaffinity="0")

    # Serialize back to XML string
    new_xml = etree.tostring(root, pretty_print=True).decode()
    file_name = "temp_scene.xml"

    # Save to temporary file
    temp_xml_path = os.path.join(os.path.dirname(xml_path), file_name)
    with open(temp_xml_path, 'w') as f:
        f.write(new_xml)

    return file_name

def add_instance(base_scene_path: str, instance_path: str, output_path: str):
    # Parse XML trees
    base_tree = etree.parse(base_scene_path)
    instance_tree = etree.parse(instance_path)

    base_root = base_tree.getroot()
    instance_root = instance_tree.getroot()

    # Locate worldbodies
    base_worldbody = base_root.find("worldbody")
    instance_worldbody = instance_root.find("worldbody")

    if base_worldbody is None or instance_worldbody is None:
        raise RuntimeError("Both base and instance XMLs must contain a <worldbody> element.")

    # Copy over all <body> elements from the instance
    for body in instance_worldbody.findall("body"):
        base_worldbody.append(copy.deepcopy(body))

    # Write the merged result
    base_tree.write(output_path, pretty_print=True)
    return output_path


def merge_robot_and_tool(
    base_dir="",
    robot_filename="ur5e.xml",
    tool_filename="small_tool.xml",
    output_robot_tool_filename="temp_ur5e_with_tool.xml",
    target_body_name="ee_frame_visual_only"
):
    robot_path = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e", robot_filename)
    tool_path = os.path.join(base_dir, "ur5e_utils_mujoco/screwing_tool", tool_filename)
    output_path = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e", output_robot_tool_filename)

    # Parse robot and tool XMLs
    robot_tree = etree.parse(robot_path)
    tool_tree = etree.parse(tool_path)
    robot_root = robot_tree.getroot()
    tool_root = tool_tree.getroot()

    # ✅ Add tool meshes and other assets to robot's <asset>
    tool_asset = tool_root.find("asset")
    if tool_asset is not None:
        robot_asset = robot_root.find("asset")
        if robot_asset is None:
            robot_asset = etree.SubElement(robot_root, "asset")
        for child in tool_asset:
            robot_asset.append(copy.deepcopy(child))

            # ✅ Copy mesh files from tool dir to robot's meshdir (ur5e/assets)
            if child.tag == "mesh" and "file" in child.attrib:
                mesh_filename = child.attrib["file"]
                src_path = os.path.join(base_dir, "ur5e_utils_mujoco/screwing_tool", mesh_filename)
                dst_path = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/assets", mesh_filename)
                if not os.path.exists(dst_path):
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copyfile(src_path, dst_path)

    # Attach only 'screw_top' body from tool
    screw_top_body = tool_root.find(".//body[@name='screw_top']")
    if screw_top_body is None:
        raise RuntimeError("Tool XML must contain a body named 'screw_top'")

    target_body = robot_root.find(f".//body[@name='{target_body_name}']")
    if target_body is None:
        raise RuntimeError(f"Target body '{target_body_name}' not found in robot XML")

    target_body.append(copy.deepcopy(screw_top_body))

    # Save merged robot+tool model
    robot_tree.write(output_path, pretty_print=True)
    return output_path


def inject_robot_tool_into_scene(
    base_dir="",
    scene_filename="empty_environment.xml",
    robot_tool_filename="temp_ur5e_with_tool.xml",
    output_scene_filename="temp_scene_with_tool.xml"
):
    scene_path = os.path.join(base_dir, "ur5e_utils_mujoco", scene_filename)
    robot_tool_path = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e", robot_tool_filename)
    output_scene_path = os.path.join(base_dir, "ur5e_utils_mujoco", output_scene_filename)

    scene_tree = etree.parse(scene_path)
    robot_tool_tree = etree.parse(robot_tool_path)
    scene_root = scene_tree.getroot()
    robot_tool_root = robot_tool_tree.getroot()

    for element in robot_tool_root:
        tag = element.tag.lower()
        if tag == "compiler":
            meshdir_rel = os.path.join("ur5e", "assets").replace("\\", "/")
            existing = scene_root.find("compiler")
            if existing is None:
                compiler = copy.deepcopy(element)
                compiler.set("meshdir", meshdir_rel)
                scene_root.insert(0, compiler)
            else:
                existing.set("meshdir", meshdir_rel)
        elif tag in ["default", "asset", "worldbody", "actuator"]:
            existing = scene_root.find(tag)
            if existing is None:
                existing = etree.SubElement(scene_root, tag)
            for child in element:
                existing.append(copy.deepcopy(child))

    scene_tree.write(output_scene_path, pretty_print=True)
    return output_scene_path


if __name__ == "__main__":

    tool_filename = "screwdriver.xml"
    robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    obstacle_name = "cube.xml"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Create the robot + tool model
    merged_robot_path = merge_robot_and_tool(tool_filename=tool_filename, base_dir=base_dir, output_robot_tool_filename=robot_and_tool_file_name)

    # Add the robot + tool to the scene
    merged_scene_path = inject_robot_tool_into_scene(robot_tool_filename=robot_and_tool_file_name, 
                                                     output_scene_filename=output_scene_filename, 
                                                     base_dir=base_dir)
    
    # Add a piece/obstacle
    obstacle_path = os.path.join(base_dir, "ur5e_utils_mujoco/obstacles", obstacle_name)
    add_instance(merged_scene_path, obstacle_path, merged_scene_path)

    # Create the reference frames
    temp_xml_name = create_reference_frames(base_dir, "ur5e_utils_mujoco/" + output_scene_filename, 1)
    model_path = os.path.join(base_dir, "ur5e_utils_mujoco", temp_xml_name)

    # Load and render
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Press ESC or Ctrl+C to exit the viewer.")
            while viewer.is_running():
                viewer.sync()
    finally:
        # Cleanup always runs
        for path in [merged_robot_path, merged_scene_path, model_path]:
            path = os.path.normpath(path)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Deleted temporary file: {path}")
                except Exception as e:
                    print(f"Could not delete {path}: {e}")
            else:
                print(f"File does not exist: {path}")

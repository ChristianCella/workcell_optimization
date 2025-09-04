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

def add_instance(base_scene_path: str, instance_path: str, output_path: str, mesh_source_dir: str = None, mesh_target_dir: str = None):
    """
    Adds <body> and <asset> elements from instance_path into base_scene_path,
    and writes the result to output_path.

    Also copies mesh files if <mesh file="..."/> is used in <asset>.

    Args:
        base_scene_path: Path to the base environment XML.
        instance_path: Path to the object/tool XML to inject.
        output_path: Path to save the modified scene XML.
        mesh_source_dir: Directory where mesh files (e.g. .obj) are located.
        mesh_target_dir: Directory where mesh files should be copied to.
    """
    base_tree = etree.parse(base_scene_path)
    instance_tree = etree.parse(instance_path)

    base_root = base_tree.getroot()
    instance_root = instance_tree.getroot()

    # --- Copy assets ---
    instance_asset = instance_root.find("asset")
    if instance_asset is not None:
        base_asset = base_root.find("asset")
        if base_asset is None:
            base_asset = etree.SubElement(base_root, "asset")
        for child in instance_asset:
            base_asset.append(copy.deepcopy(child))

            # --- Copy mesh files if needed ---
            if child.tag == "mesh" and "file" in child.attrib and mesh_source_dir and mesh_target_dir:
                mesh_file = child.attrib["file"]
                src = os.path.join(mesh_source_dir, mesh_file)
                dst = os.path.join(mesh_target_dir, mesh_file)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    try:
                        shutil.copyfile(src, dst)
                        print(f"Copied mesh: {src} -> {dst}")
                    except Exception as e:
                        print(f"Failed to copy mesh {mesh_file}: {e}")

    # Copy <default> section (optional)
    instance_default = instance_root.find("default")
    if instance_default is not None:
        base_default = base_root.find("default")
        if base_default is None:
            base_default = etree.SubElement(base_root, "default")
        for child in instance_default:
            base_default.append(copy.deepcopy(child))

    # Copy <body> from <worldbody>
    base_worldbody = base_root.find("worldbody")
    instance_worldbody = instance_root.find("worldbody")
    if base_worldbody is None or instance_worldbody is None:
        raise RuntimeError("Missing <worldbody> in base or instance XML.")

    for body in instance_worldbody.findall("body"):
        base_worldbody.append(copy.deepcopy(body))

    # Save the updated XML
    base_tree.write(output_path, pretty_print=True)
    return output_path


def merge_robot_and_tool(
    base_dir="",
    robot_filename="iiwa14.xml",
    tool_filename="small_tool.xml",
    output_robot_tool_filename="temp_kuka_with_tool.xml",
    target_body_name="ee_frame_visual_only"
):
    robot_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/kuka", robot_filename)
    tool_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/screwing_tool", tool_filename)
    output_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/kuka", output_robot_tool_filename)

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
                src_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/screwing_tool", mesh_filename)
                dst_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/kuka/assets", mesh_filename)
                if not os.path.exists(dst_path):
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copyfile(src_path, dst_path)

    # Attach only 'tool_top' body from tool
    tool_top_body = tool_root.find(".//body[@name='tool_top']")
    if tool_top_body is None:
        raise RuntimeError("Tool XML must contain a body named 'tool_top'")

    target_body = robot_root.find(f".//body[@name='{target_body_name}']")
    if target_body is None:
        raise RuntimeError(f"Target body '{target_body_name}' not found in robot XML")

    target_body.append(copy.deepcopy(tool_top_body))

    # Save merged robot+tool model
    robot_tree.write(output_path, pretty_print=True)
    return output_path

def add_contact_exclusions(root):
    # Ensure the <contact> section exists
    contact = root.find("contact")
    if contact is None:
        contact = etree.SubElement(root, "contact")

    exclusions = [
        ("base", "link1"),
        ("base", "link2"),
        ("base", "link3"),
        ("link1", "link3"),
        ("link3", "link5"),
        ("link4", "link7"),
        ("link5", "link7"),
    ]

    for body1, body2 in exclusions:
        etree.SubElement(contact, "exclude", body1=body1, body2=body2)



def inject_robot_tool_into_scene(
    base_dir="",
    scene_filename="empty_environment.xml",
    robot_tool_filename="temp_kuka_with_tool.xml",
    output_scene_filename="temp_scene_with_tool.xml"
):
    scene_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils", scene_filename)
    robot_tool_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/kuka", robot_tool_filename)
    output_scene_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils", output_scene_filename)

    scene_tree = etree.parse(scene_path)
    robot_tool_tree = etree.parse(robot_tool_path)
    scene_root = scene_tree.getroot()
    robot_tool_root = robot_tool_tree.getroot()

    for element in robot_tool_root:
        tag = element.tag.lower()
        if tag == "compiler":
            meshdir_rel = os.path.join("kuka", "assets").replace("\\", "/")
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

def create_scene(tool_name, robot_and_tool_file_name, output_scene_filename, piece_name1, piece_name2, base_dir):

    # Create the robot + tool model
    _ = merge_robot_and_tool(tool_filename=tool_name, base_dir=base_dir, output_robot_tool_filename=robot_and_tool_file_name)

    # Add the robot + tool to the 'vanilla' scene
    merged_scene_path = inject_robot_tool_into_scene(robot_tool_filename=robot_and_tool_file_name, 
                                                     output_scene_filename=output_scene_filename, 
                                                     base_dir=base_dir)
    
    # Add another instance (i.e. piece for screwing, cockpit)
    obstacle_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/screwing_pieces", piece_name1)
    add_instance(
        merged_scene_path,
        obstacle_path,
        merged_scene_path,
        mesh_source_dir=os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/screwing_pieces"),
        mesh_target_dir=os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/kuka/assets")
    )

    # Add another instance (i.e. piece for screwing, cockpit)
    obstacle_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/simple_obstacles", piece_name2)
    add_instance(
        merged_scene_path,
        obstacle_path,
        merged_scene_path,
        mesh_source_dir=os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/screwing_pieces"),
        mesh_target_dir=os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/kuka/assets")
    )

    # Define the path to the final scene
    model_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils", output_scene_filename)

    tree = etree.parse(model_path)
    root = tree.getroot()
    add_contact_exclusions(root)
    tree.write(model_path, pretty_print=True)

    return model_path


if __name__ == "__main__":

    tool_filename = "driller.xml"
    robot_and_tool_file_name = "temp_kuka_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    piece_name1 = "aluminium_plate.xml" 
    piece_name2 = "Linear_guide.xml"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Create the method defined above
    model_path = create_scene(tool_name=tool_filename,
                              robot_and_tool_file_name=robot_and_tool_file_name,
                              output_scene_filename=output_scene_filename,
                              piece_name1=piece_name1,
                              piece_name2=piece_name2,
                              base_dir=base_dir)

    # Load and render
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Press ESC or Ctrl+C to exit the viewer.")
        while viewer.is_running():
            viewer.sync()


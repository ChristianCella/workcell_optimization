from lxml import etree
import mujoco
import mujoco.viewer
import sys
import os
import time

def create_reference_frames(base_dir, vanilla_scene, n_targets):
    """
    Loads a MuJoCo scene, adds a new reference frame to it,
    saves it temporarily, and opens it in a passive viewer.
    """
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

    # Save to temporary file
    temp_xml_path = os.path.join(os.path.dirname(xml_path), "temp_scene.xml")
    with open(temp_xml_path, 'w') as f:
        f.write(new_xml)

    return temp_xml_path


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    temp_xml_path = create_reference_frames(base_dir, "ur5e_utils_mujoco/scene.xml", 1)

    # Load model from the temporary file
    model = mujoco.MjModel.from_xml_path(temp_xml_path)
    data = mujoco.MjData(model)

    # Initialize simulation state
    mujoco.mj_forward(model, data)

    # View with minimal updates (static visualization)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        target_fps = 30
        frame_time = 1.0 / target_fps

        while viewer.is_running():
            step_start = time.time()
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

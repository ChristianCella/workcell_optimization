from lxml import etree
import mujoco
import mujoco.viewer
import sys, os
import time

''' 
This code allows to create 'dynamically' teh frames needed for the scene.
It craetes a copy of the original scene.xml file with the requested modifications.
Sometimes mujoco can be 'laggy' ,unless you specirfy a suitable refresh rate.
'''

# Setup
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_dir)
xml_path = os.path.join(base_dir, "ur5e_utils_mujoco/scene.xml")

# Load the model first with MuJoCo to resolve includes
original_model = mujoco.MjModel.from_xml_path(xml_path)

# Now load and modify the XML
with open(xml_path, 'r') as f:
    xml_string = f.read()

# Parse and modify
root = etree.fromstring(xml_string)

# Add a new frame body
worldbody = root.find(".//worldbody")
body = etree.SubElement(worldbody, "body", name="reference_target_10", pos="0.1 0.1 0.1")
etree.SubElement(body, "geom", name="reference_x", type="capsule", fromto="0 0 0 0.1 0 0", size="0.005", rgba="1 0 0 0.5", contype="0", conaffinity="0")
etree.SubElement(body, "geom", name="reference_y", type="capsule", fromto="0 0 0 0 0.1 0", size="0.005", rgba="0 1 0 0.5", contype="0", conaffinity="0")
etree.SubElement(body, "geom", name="reference_z", type="capsule", fromto="0 0 0 0 0 0.1", size="0.005", rgba="0 0 1 0.5", contype="0", conaffinity="0")

# Serialize back to XML string
new_xml = etree.tostring(root, pretty_print=True).decode()

# Save to temporary file in the same directory as the original
temp_xml_path = os.path.join(os.path.dirname(xml_path), "temp_scene.xml")
with open(temp_xml_path, 'w') as f:
    f.write(new_xml)

try:
    # Load model from the temporary file (includes will resolve correctly)
    model = mujoco.MjModel.from_xml_path(temp_xml_path)
    data = mujoco.MjData(model)

    # Initialize the simulation state once
    mujoco.mj_forward(model, data)

    # View with minimal updates (static visualization)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        target_fps = 30  # Lower FPS for static content
        frame_time = 1.0 / target_fps
        
        while viewer.is_running():
            step_start = time.time()
            
            # Only sync viewer, no physics stepping
            viewer.sync()
            
            # Frame rate control
            elapsed = time.time() - step_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

finally:
    # Clean up temporary file
    if os.path.exists(temp_xml_path):
        os.remove(temp_xml_path)
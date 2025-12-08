import mujoco
import numpy as np
from lxml import etree
import sys, os
import copy
from transformations import rotm_to_quaternion, quaternion_to_euler

def set_body_pose(model, data, body_id, pos, quat):
    model.body_pos[body_id] = pos
    model.body_quat[body_id] = quat
    mujoco.mj_forward(model, data)

def get_cartesian_pose(frame_id, data, representation):
    position = data.xpos[frame_id]
    rotation_matrix = data.xmat[frame_id].reshape(3, 3)
    quaternion = rotm_to_quaternion(rotation_matrix)
    euler_angles = quaternion_to_euler(quaternion, degrees=False)
    if representation == "quaternion":
        return position, quaternion
    elif representation == "euler":
        return position, euler_angles
    elif representation == "rotation_matrix":
        return position, rotation_matrix

def compute_jacobian(model, data, tool_site_id):
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
    Jac = np.vstack([Jp, Jr])[:, :6]
    return Jac

def get_collisions(model, data, verbose):
    # Step the simulator once so that contacts get populated
    mujoco.mj_forward(model, data)

    if data.ncon == 0:
        if verbose: print("No collisions detected.")
    else:
        if verbose: print(f"{data.ncon} collision(s) detected:")
        for i in range(data.ncon):
            c = data.contact[i]
            # lookup names via mj_id2name
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            if verbose: print(f"  • {name1} ↔ {name2}")
    return data.ncon

def inverse_manipulability(q, model, data, tool_site_id):
    data.qpos[:model.nv] = q; mujoco.mj_forward(model, data)
    J = compute_jacobian(model, data, tool_site_id)
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return 1e12 if det <= 1e-12 else 1.0/np.sqrt(det)

def create_reference_frames(starting_scene_path, n_targets, final_xml_directory):
    # ! Load and modify the xml
    xml_path = starting_scene_path
    with open(xml_path, 'r') as f:
        xml_string = f.read()

    # Parse the xml
    root = etree.fromstring(xml_string)

    # Add all the n_targets reference frames
    worldbody = root.find(".//worldbody")
    for i in range(n_targets):
        target_name = f"reference_target_{i+1}"
        x_axis_name = f"reference_x_axis{i+1}"
        y_axis_name = f"reference_y_axis{i+1}"
        z_axis_name = f"reference_z_axis{i+1}"
        body = etree.SubElement(worldbody, "body", name=target_name, pos="0.0 0.0 0.0")
        etree.SubElement(body, "geom", name=x_axis_name, type="capsule", fromto="0 0 0 0.1 0 0", size="0.008", rgba="1 0 0 0.5", contype="0", conaffinity="0")
        etree.SubElement(body, "geom", name=y_axis_name, type="capsule", fromto="0 0 0 0 0.1 0", size="0.008", rgba="0 1 0 0.5", contype="0", conaffinity="0")
        etree.SubElement(body, "geom", name=z_axis_name, type="capsule", fromto="0 0 0 0 0 0.1", size="0.008", rgba="0 0 1 0.5", contype="0", conaffinity="0")

    # Serialize back to XML string
    new_xml = etree.tostring(root, pretty_print=True).decode()
    file_name = "temp_scene.xml"

    # Save to temporary file
    temp_xml_path = os.path.join(final_xml_directory, file_name)
    with open(temp_xml_path, 'w') as f:
        f.write(new_xml)

    return temp_xml_path


def add_extension_to_scene(starting_scene_path, extension_xml_path, final_xml_directory):
    """
    Merge an 'extension' MJCF into a starting scene MJCF.

    - Imports the meshes defined in extension_xml_path into the <asset> section
    - Removes 'assets/' prefix from mesh file paths
    - Imports the main body from extension_xml_path's <worldbody>
    - Applies extension <default><geom> attributes to imported geoms
    - Saves a new XML in final_xml_directory and returns its path
    """

    # --- Load and parse the starting scene ---
    with open(starting_scene_path, "r") as f:
        starting_xml_string = f.read()
    starting_root = etree.fromstring(starting_xml_string)

    # --- Load and parse the extension file ---
    with open(extension_xml_path, "r") as f:
        extension_xml_string = f.read()
    extension_root = etree.fromstring(extension_xml_string)

    # -------------------------------------------------------------------------
    # 1) Merge assets (meshes etc.)
    # -------------------------------------------------------------------------
    starting_asset = starting_root.find(".//asset")
    if starting_asset is None:
        starting_asset = etree.SubElement(starting_root, "asset")

    extension_asset = extension_root.find(".//asset")
    if extension_asset is not None:
        existing_assets = {
            (child.tag, child.get("name"))
            for child in starting_asset
            if child.get("name") is not None
        }

        for ext_child in extension_asset:
            key = (ext_child.tag, ext_child.get("name"))
            if ext_child.get("name") is not None and key in existing_assets:
                continue

            # -----------------------------
            # ✨ REMOVE "assets/" prefix
            # -----------------------------
            if ext_child.tag == "mesh" and ext_child.get("file"):
                file_path = ext_child.get("file")
                if file_path.startswith("assets/"):
                    # keep only the filename
                    ext_child.set("file", file_path.replace("assets/", "", 1))
            # -----------------------------

            starting_asset.append(copy.deepcopy(ext_child))

    # -------------------------------------------------------------------------
    # 2) Apply default geom settings to imported geoms
    # -------------------------------------------------------------------------
    extension_default_geom = extension_root.find(".//default/geom")
    default_geom_attrs = {}
    if extension_default_geom is not None:
        default_geom_attrs = dict(extension_default_geom.attrib)

    # -------------------------------------------------------------------------
    # 3) Copy extension body into the starting scene worldbody
    # -------------------------------------------------------------------------
    starting_worldbody = starting_root.find(".//worldbody")
    extension_worldbody = extension_root.find(".//worldbody")

    extension_bodies = list(extension_worldbody.findall("./body"))
    ext_base_body = extension_bodies[0]
    ext_base_copy = copy.deepcopy(ext_base_body)

    # Apply imported geom defaults
    if default_geom_attrs:
        for geom in ext_base_copy.iter("geom"):
            for key, val in default_geom_attrs.items():
                if geom.get(key) is None:
                    geom.set(key, val)

    starting_worldbody.append(ext_base_copy)

    # -------------------------------------------------------------------------
    # 4) Save updated XML
    # -------------------------------------------------------------------------
    new_xml = etree.tostring(starting_root, pretty_print=True).decode()
    file_name = "temp_scene.xml"
    temp_xml_path = os.path.join(final_xml_directory, file_name)

    with open(temp_xml_path, "w") as f:
        f.write(new_xml)

    return temp_xml_path

def scene_manager(case_id, number_targets, base_dir, robot_name, extension_name):
    if case_id == "robot":
        model_path = os.path.join(base_dir, robot_name)
        return model_path
    elif case_id == "targets":
        model_path = os.path.join(base_dir, robot_name)
        model_path = create_reference_frames(model_path, number_targets, base_dir)
        return model_path
    elif case_id == "full":
        model_path = os.path.join(base_dir, robot_name)
        model_path = create_reference_frames(model_path, number_targets, base_dir)
        model_path = add_extension_to_scene(model_path, os.path.join(base_dir, extension_name), base_dir)
        return model_path





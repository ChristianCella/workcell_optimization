import mariadb

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'user',
    'password': 'Robotics.123456',
    'database': 'ARTO'
}

def get_atomic_by_name(atomic_name):
    conn = mariadb.connect(**DB_CONFIG)
    cursor = conn.cursor()
    query = """SELECT offset_pose, task_frame FROM robotic_atomics WHERE name = %s"""
    cursor.execute(query, (atomic_name,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result:
        raise ValueError(f"No RoboticAtomic found with name: {atomic_name}")

    offset_pose = result[0].strip('{}')
    offset_values = [float(x) for x in offset_pose.split(',')]
    if len(offset_values) != 7:
        raise ValueError(f"Invalid offset_pose size for: {atomic_name}")

    task_frame_id = result[1]
    return offset_values, task_frame_id

def get_panel_by_ID(ID):
    conn = mariadb.connect(**DB_CONFIG)
    cursor = conn.cursor()
    query = """SELECT panel_belonging FROM features WHERE id = %s"""
    cursor.execute(query, (ID,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result:
        raise ValueError(f"No feature found with ID: {ID}")

    panel_belonging = result[0]
    return panel_belonging

def get_panel_pose(panel_name):
    conn = mariadb.connect(**DB_CONFIG)
    cursor = conn.cursor()
    query = """SELECT coordinates_wrt_robot FROM panels WHERE name = %s"""
    cursor.execute(query, (panel_name,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result:
        raise ValueError(f"No panel found with name: {panel_name}")

    panel_pose = result[0].strip('{}')
    return panel_pose

if __name__ == "__main__":
    atomic_name = "MRfcuappr" # Row 4 of the table 'robotic_atomics'
    # Example execution
    try:
        offset_values, task_frame_id = get_atomic_by_name(atomic_name)
        print(f"Offset: {offset_values}, Task Frame ID: {task_frame_id}")
        panel_name = get_panel_by_ID(task_frame_id) # Use the ID to get the name of the panel from the table 'features'
        print(f"Panel Name: {panel_name}")
        panel_pose = get_panel_pose(panel_name)
        print(f"Panel Pose wrt robot: {panel_pose}")
    except Exception as e:
        print(f"Error: {e}")



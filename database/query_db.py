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

# Example execution
try:
    offset_values, task_frame_id = get_atomic_by_name("MRfcuappr")
    print(f"Offset: {offset_values}, Task Frame ID: {task_frame_id}")
except Exception as e:
    print(f"Error: {e}")

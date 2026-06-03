import numpy as np

R = 0.03 # meters
#thetas = [120, 150, 180, 210, 240]
thetas = [60, 30, 0, -30, -60]
px = 0.1
py = 0.03
for theta in thetas:
    x = px + R * np.cos(np.radians(theta))
    y = py + R * np.sin(np.radians(theta))
    print(f"theta: {theta} -> x: {x:.3f}, y: {y:.3f}")
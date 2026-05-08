import random, math
import statistics

from selfcal_omr.helper import Point

# generate random points
random.seed(0)


def dist2(a,b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def are_collinear(p, p1, p2, tol=1e-1):
    # area of triangle formula ~ 0 for collinear
    area = abs(p1[0]*(p2[1]-p[1]) + p2[0]*(p[1]-p1[1]) + p[0]*(p1[1]-p2[1])) / 2
    return area < tol

def angle_approx_90(p, p1, p2, tol_deg=5):
    # vectors
    v1 = (p1[0]-p[0], p1[1]-p[1])
    v2 = (p2[0]-p[0], p2[1]-p[1])

    # dot product
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return False

    # angle
    angle = math.degrees(math.acos(max(-1, min(1, dot/(mag1*mag2)))))
    return abs(angle - 90) <= tol_deg

def do(points, n, tol = 40):
    s =0 
    if len(points) <= 3:
        return 0,0
    random.seed(0)
    if len(points) <= 50:
        samples = points
    else: 
        samples = random.sample(points, n)
    results = []
    for p in samples:
        # pick random p
        #p = random.choice(points)

        # find two nearest points to p
        d_sorted = sorted(points, key=lambda q: dist2(p, q))
        p1, p2 = d_sorted[1], d_sorted[2] # 0 is p itself
        
        if are_collinear(p, p1, p2, tol):
            dx = p[0] - p2[0]
            dy = p[1] - p2[1]
            if dx == 0:
                if dy > 0:
                    s+=1
                    results.append(90)
                elif dy < 0:
                    s+=1
                    results.append(-90)
                continue
            s+=1
            val = math.atan(dy/dx)
            results.append(val*180/math.pi)
            continue
        # check approx right angle
        if not angle_approx_90(p, p1, p2):
            continue
        

        # compute arctan((p.y - p2.y)/(p.x - p2.x))
        dx = p[0] - p1[0]
        dy = p[1] - p1[1]
        if dx == 0:
            if dy > 0:
                results.append(90)
            elif dy < 0:
                results.append(-90)

        val = math.atan(dy/dx)
        results.append(val*180/math.pi)

    mean_val = statistics.median(results) if results else None
    return mean_val, s#len(results)

import numpy as np

# 1. Create a 10x10 grid
rows, cols = 10, 10
grid = [(10 * x, 15 * y) for y in range(rows) for x in range(cols)]

# Convert to numpy array
points = np.array(grid)

# 2. Rotation angle
theta = np.deg2rad(45)

# Rotation matrix
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# 3. Rotate the grid
rotated_points = points @ R.T

def Rotate(points, center, deg):
    ps = [p.to_numpy() - center for p in points]
    theta = np.deg2rad(deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    rotated_points =  ps @ R.T
    return [Point(p[0] + center[0], p[1] + center[1]) for p in rotated_points]

def Rotate_(point, center, deg):
    p_= point.to_numpy() - center
    theta = np.deg2rad(deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    p =  R @ p_.T
    return Point(p[0] + center[0], p[1] + center[1])
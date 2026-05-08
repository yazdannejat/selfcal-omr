# Optimized DBSCAN in pure Python (grid‑accelerated)

import math
from collections import defaultdict

def dbscan(points, eps, min_pts):
    eps2 = eps * eps      # squared eps
    grid = defaultdict(list)

    # ----- Step 1: Build spatial grid -----
    # cell size = eps, so only nearby cells must be checked
    cell_size = eps

    def cell_of(p):
        return (int(p[0] // cell_size), int(p[1] // cell_size))

    for i, p in enumerate(points):
        grid[cell_of(p)].append(i)

    # Precompute the 9 neighbor cells (3×3 region)
    neighbor_cells = [(dx, dy) for dx in (-1, 0, 1)
                               for dy in (-1, 0, 1)]

    # ----- Helper: region query using grid -----
    def region_query(idx):
        px, py = points[idx]
        cx, cy = cell_of((px, py))

        neighbors = []
        for dx, dy in neighbor_cells:
            cell = (cx + dx, cy + dy)
            if cell in grid:
                for j in grid[cell]:
                    qx, qy = points[j]
                    # squared distance check
                    if (px - qx) * (px - qx) + (py - qy) * (py - qy) <= eps2:
                        neighbors.append(j)
        return neighbors

    # ----- Expand cluster -----
    def expand_cluster(idx, neighbors, cluster_id):
        labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            n_idx = neighbors[i]

            if labels[n_idx] == -1:
                labels[n_idx] = cluster_id  # noise → border point

            elif labels[n_idx] == 0:
                labels[n_idx] = cluster_id

                n_neighbors = region_query(n_idx)
                if len(n_neighbors) >= min_pts:
                    neighbors.extend(n_neighbors)

            i += 1

    # ----- Main loop -----
    labels = [0] * len(points)
    cluster_id = 0

    for idx in range(len(points)):
        if labels[idx] != 0:
            continue

        neighbors = region_query(idx)

        if len(neighbors) < min_pts:
            labels[idx] = -1
        else:
            cluster_id += 1
            expand_cluster(idx, neighbors, cluster_id)

    return labels

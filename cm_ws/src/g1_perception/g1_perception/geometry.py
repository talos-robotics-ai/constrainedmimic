"""Point labelling and box fitting for scene_capture (numpy only, no ROS).

All points are in the depth optical frame (x right, y down, z forward). A box is
(centre, rotation, half_extents): rotation columns are the box axes, half_extents
the half sizes along them.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull


def depth_to_points(depth_m, k):
    """(H, W) depth in metres + 3x3 intrinsics -> (H, W, 3) points, z = 0 where invalid."""
    height, width = depth_m.shape
    v, u = np.mgrid[0:height, 0:width]
    x = (u - k[0, 2]) / k[0, 0] * depth_m
    y = (v - k[1, 2]) / k[1, 1] * depth_m
    return np.stack([x, y, depth_m], axis=-1)


def label_points(points, rotation, translation, k_color, label_image):
    """Label each depth point with the label_image value at its colour pixel (0 = none).

    rotation, translation map depth-frame points into the colour frame. Points
    behind the colour camera or outside its image get 0.
    """
    p = points @ rotation.T + translation
    labels = np.zeros(len(points), label_image.dtype)
    in_front = p[:, 2] > 1e-3
    u = np.round(k_color[0, 0] * p[in_front, 0] / p[in_front, 2] + k_color[0, 2]).astype(int)
    v = np.round(k_color[1, 1] * p[in_front, 1] / p[in_front, 2] + k_color[1, 2]).astype(int)
    height, width = label_image.shape
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    idx = np.flatnonzero(in_front)[inside]
    labels[idx] = label_image[v[inside], u[inside]]
    return labels


def largest_cluster(points, voxel):
    """Points of the largest 26-connected voxel cluster (drops mask edges bleeding onto
    the background, which sit at a different depth)."""
    if len(points) == 0:
        return points
    cells = np.floor((points - points.min(axis=0)) / voxel).astype(int)
    grid = np.zeros(cells.max(axis=0) + 1, bool)
    grid[tuple(cells.T)] = True
    components, count = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if count <= 1:
        return points
    point_component = components[tuple(cells.T)]
    largest = np.argmax(np.bincount(point_component)[1:]) + 1
    return points[point_component == largest]


def fit_plane(points, threshold, iterations=200, rng=None):
    """RANSAC plane -> (normal, offset, inlier mask) with normal . p + offset = 0.

    The normal is refined on the inliers and points towards the camera (the origin),
    i.e. up out of a table seen from above.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    # Score the hypotheses on a subsample; a table mask can hold 100k+ points
    sample = points[rng.choice(len(points), min(len(points), 20000), replace=False)]
    best = np.zeros(len(sample), bool)
    for _ in range(iterations):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        inliers = np.abs((sample - a) @ normal) < threshold
        if inliers.sum() > best.sum():
            best = inliers
    centroid = sample[best].mean(axis=0)
    normal = np.linalg.svd(sample[best] - centroid, full_matrices=False)[2][2]
    offset = -normal @ centroid
    if offset < 0:
        normal, offset = -normal, -offset
    return normal, offset, np.abs(points @ normal + offset) < threshold


def plane_rotation(normal):
    """Rotation whose z axis is the normal and x axis the camera x projected onto the plane."""
    x = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    if np.linalg.norm(x) < 1e-6:  # normal along camera x
        x = np.array([0.0, 1.0, 0.0]) - normal[1] * normal
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(normal, x), normal])


def min_area_rect(xy):
    """Smallest rotated rectangle around 2D points -> (centre, half sizes, angle)."""
    hull = xy[ConvexHull(xy).vertices] if len(xy) >= 3 else xy
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    best = None
    for angle in np.arctan2(edges[:, 1], edges[:, 0]):
        c, s = np.cos(angle), np.sin(angle)
        local = hull @ np.array([[c, -s], [s, c]])
        lo, hi = local.min(axis=0), local.max(axis=0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, angle, lo, hi)
    _, angle, lo, hi = best
    c, s = np.cos(angle), np.sin(angle)
    centre = np.array([[c, -s], [s, c]]) @ ((lo + hi) / 2)
    return centre, (hi - lo) / 2, angle


def box_on_plane(points, plane_rot, plane_offset, z_range):
    """Box standing on a plane: footprint = min-area rectangle of the points projected on
    the plane, height = z_range (bottom, top) in metres along the plane normal, measured
    from the plane."""
    local = points @ plane_rot  # coordinates in the plane frame
    plane_z = -plane_offset  # plane height along its own normal
    centre_xy, half_xy, angle = min_area_rect(local[:, :2])
    c, s = np.cos(angle), np.sin(angle)
    rotation = plane_rot @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    bottom, top = z_range
    centre = plane_rot @ np.array([centre_xy[0], centre_xy[1], plane_z + (bottom + top) / 2])
    return centre, rotation, np.array([half_xy[0], half_xy[1], (top - bottom) / 2])


def box_pca(points):
    """Oriented bounding box from the principal axes (no support plane known)."""
    centroid = points.mean(axis=0)
    axes = np.linalg.svd(points - centroid, full_matrices=False)[2].T
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]
    local = (points - centroid) @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    return centroid + axes @ ((lo + hi) / 2), axes, (hi - lo) / 2


def heights_above(points, plane_rot, plane_offset):
    """Signed distance of points above the plane (along its normal)."""
    return points @ plane_rot[:, 2] + plane_offset

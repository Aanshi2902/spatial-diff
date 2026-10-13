"""3D change detection with free-space reasoning.

A voxel counts as ADDED only if it is occupied in scene B AND scene A's camera
rays passed through it (observed empty). A voxel never observed in A is
UNKNOWN and is ignored. This is what prevents occlusion and coverage gaps from
being reported as changes.

Scene format (dict):
    points      (N,3) fused point cloud, already in the shared coordinate frame
    cams        list of (center(3,), frame_points(M,3)) one entry per camera frame
    colors      (N,3) optional, 0..1
"""
import numpy as np
from scipy import ndimage

STRUCT26 = ndimage.generate_binary_structure(3, 3)


class Grid:
    def __init__(self, lo, vox, dims):
        self.lo, self.vox, self.dims = lo, vox, dims

    def ijk(self, p):
        i = np.floor((p - self.lo) / self.vox).astype(int)
        return np.clip(i, 0, np.array(self.dims) - 1)

    def center(self, ijk):
        return self.lo + (np.asarray(ijk) + 0.5) * self.vox


def build_grid(point_sets, vox, pad=3):
    allp = np.vstack(point_sets)
    lo = allp.min(0) - pad * vox
    hi = allp.max(0) + pad * vox
    dims = tuple(np.ceil((hi - lo) / vox).astype(int) + 1)
    return Grid(lo, vox, dims)


def occupancy(points, grid, min_pts=2):
    idx = grid.ijk(points)
    flat = np.ravel_multi_index(idx.T, grid.dims)
    counts = np.bincount(flat, minlength=int(np.prod(grid.dims)))
    return (counts >= min_pts).reshape(grid.dims)


def free_space(cams, grid, max_pts=6000, margin=2.5, seed=0):
    """Mark voxels that camera rays traversed before hitting a surface."""
    rng = np.random.default_rng(seed)
    free = np.zeros(grid.dims, bool)
    v = grid.vox
    for center, pts in cams:
        if len(pts) > max_pts:
            pts = pts[rng.choice(len(pts), max_pts, replace=False)]
        d = pts - center
        dist = np.linalg.norm(d, axis=1)
        keep = dist > margin * v
        d, dist = d[keep], dist[keep]
        if len(d) == 0:
            continue
        dirn = d / dist[:, None]
        k = int(dist.max() / (0.7 * v)) + 2
        u = np.linspace(0, 1, k)
        for s in range(0, len(d), 2000):
            sl = slice(s, s + 2000)
            reach = np.maximum(dist[sl] - margin * v, 0)
            pos = center + dirn[sl][:, None, :] * (u[None, :, None] * reach[:, None, None])
            ii = grid.ijk(pos.reshape(-1, 3))
            free[ii[:, 0], ii[:, 1], ii[:, 2]] = True
    return free


def _components(mask, min_voxels):
    lab, n = ndimage.label(mask, structure=STRUCT26)
    out = []
    for k in range(1, n + 1):
        ijk = np.argwhere(lab == k)
        if len(ijk) >= min_voxels:
            out.append(ijk)
    return out


def _cluster_info(kind, ijk, grid, scene_pts, scene_cols):
    world = grid.center(ijk)
    lo = world.min(0) - grid.vox / 2
    hi = world.max(0) + grid.vox / 2
    inside = np.all((scene_pts >= lo) & (scene_pts <= hi), axis=1)
    color = None
    if scene_cols is not None and inside.any():
        color = scene_cols[inside].mean(0)
    return dict(kind=kind, n_voxels=int(len(ijk)), center=world.mean(0),
                bbox_min=lo, bbox_max=hi, extent=hi - lo,
                n_points=int(inside.sum()), color=color)


def detect(A, B, vox, tol=1, min_voxels=8, min_pts=2):
    """Return dict with 'added', 'removed' cluster lists and diagnostics."""
    grid = build_grid([A["points"], B["points"]], vox)
    occA = occupancy(A["points"], grid, min_pts)
    occB = occupancy(B["points"], grid, min_pts)
    near_occA = ndimage.binary_dilation(occA, STRUCT26, iterations=tol)
    near_occB = ndimage.binary_dilation(occB, STRUCT26, iterations=tol)
    freeA = free_space(A["cams"], grid) & ~near_occA
    freeB = free_space(B["cams"], grid) & ~near_occB

    added = occB & freeA
    removed = occA & freeB
    unknown_removed = occA & ~freeB & ~near_occB   # occupied in A, B never looked

    res = dict(grid=grid, added=[], removed=[],
               n_unknown_voxels=int(unknown_removed.sum()))
    for ijk in _components(added, min_voxels):
        res["added"].append(_cluster_info("added", ijk, grid, B["points"], B.get("colors")))
    for ijk in _components(removed, min_voxels):
        res["removed"].append(_cluster_info("removed", ijk, grid, A["points"], A.get("colors")))
    return res


def naive_diff(A, B, vox, dist_thresh, min_voxels=8):
    """Baseline for the ablation: plain nearest-neighbour point distance."""
    from scipy.spatial import cKDTree
    grid = build_grid([A["points"], B["points"]], vox)
    dB = cKDTree(A["points"]).query(B["points"])[0]
    dA = cKDTree(B["points"]).query(A["points"])[0]
    added = occupancy(B["points"][dB > dist_thresh], grid, 1)
    removed = occupancy(A["points"][dA > dist_thresh], grid, 1)
    out = dict(added=[], removed=[])
    for ijk in _components(added, min_voxels):
        out["added"].append(_cluster_info("added", ijk, grid, B["points"], B.get("colors")))
    for ijk in _components(removed, min_voxels):
        out["removed"].append(_cluster_info("removed", ijk, grid, A["points"], A.get("colors")))
    return out

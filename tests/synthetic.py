"""Synthetic rooms: ray-cast two scenes from two camera paths, so occlusion is real.

Scene A -> Scene B changes (ground truth):
    MOVED   chair       (-0.55, 0.55) -> (0.45, -0.55)
    ADDED   backpack    near (-0.6, -1.2)
    REMOVED box         near (-1.6, -0.2)
Camera path B is shorter than A, so part of A is never seen in B.
"""
import numpy as np

GT = [
    dict(kind="MOVED", label="chair", center=(0.45, -0.55, 0.45)),
    dict(kind="ADDED", label="backpack", center=(-0.6, -1.2, 0.25)),
    dict(kind="REMOVED", label="box", center=(-1.6, -0.2, 0.15)),
]

STATIC = [
    ((-3, -3, -0.05), (3, 3, 0.0)),        # floor
    ((-3, 3, 0), (3, 3.05, 2.5)),          # back wall
    ((-3.05, -3, 0), (-3, 3, 2.5)),        # left wall
    ((3, -3, 0), (3.05, 3, 2.5)),          # right wall
    ((0.5, 1.0, 0), (1.7, 1.8, 0.75)),     # desk
    ((-2.2, 1.5, 0), (-1.4, 2.4, 1.2)),    # shelf (visible mostly from the left)
]
CHAIR_A = ((-0.8, 0.3, 0), (-0.3, 0.8, 0.9))
CHAIR_B = ((0.2, -0.8, 0), (0.7, -0.3, 0.9))
BOX_A = ((-1.8, -0.4, 0), (-1.4, 0.0, 0.3))
BACKPACK_B = ((-0.75, -1.3, 0), (-0.45, -1.1, 0.5))


COLORS = {0: (0.55, 0.55, 0.55), 1: (0.8, 0.8, 0.75), 2: (0.75, 0.8, 0.8), 3: (0.8, 0.75, 0.8),
          4: (0.55, 0.35, 0.2), 5: (0.3, 0.3, 0.35)}  # floor, walls, desk, shelf
OBJ_COLORS = {"chair": (0.15, 0.3, 0.85), "box": (0.85, 0.2, 0.15), "backpack": (0.1, 0.6, 0.25)}


def scene_boxes(which):
    if which == "A":
        return STATIC + [CHAIR_A, BOX_A]
    return STATIC + [CHAIR_B, BACKPACK_B]


def box_colors(which):
    objs = ["chair", "box"] if which == "A" else ["chair", "backpack"]
    return [COLORS.get(i, (0.6, 0.6, 0.6)) for i in range(len(STATIC))] + [OBJ_COLORS[o] for o in objs]


def raycast(origin, dirs, boxes):
    best = np.full(len(dirs), np.inf)
    which_box = np.full(len(dirs), -1)
    dirs = np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)
    for bi, (lo, hi) in enumerate(boxes):
        t1 = (np.array(lo) - origin) / dirs
        t2 = (np.array(hi) - origin) / dirs
        tmin = np.minimum(t1, t2).max(1)
        tmax = np.maximum(t1, t2).min(1)
        hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
        upd = hit & (tmin < best)
        best = np.where(upd, tmin, best)
        which_box = np.where(upd, bi, which_box)
    return best, which_box


def camera_dirs(origin, target, w=80, h=60, fov=75):
    f = (target - origin); f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    xs = np.tan(np.radians(fov / 2)) * np.linspace(-1, 1, w)
    ys = np.tan(np.radians(fov / 2)) * np.linspace(-1, 1, h) * h / w
    X, Y = np.meshgrid(xs, ys)
    d = f[None] + X.reshape(-1, 1) * r[None] + Y.reshape(-1, 1) * u[None]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def make_scene(which, arc, n_cams=30, radius=2.4, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    boxes = scene_boxes(which)
    center = np.array([0.0, 0.5, 0.0])
    cams, all_pts, all_cols = [], [], []
    palette = np.array(box_colors(which))
    for th in np.radians(np.linspace(arc[0], arc[1], n_cams)):
        o = center + [radius * np.cos(th), radius * np.sin(th), 1.3]
        d = camera_dirs(o, center + [0, 0, 0.6])
        t, wb = raycast(o, d, boxes)
        ok = np.isfinite(t) & (t < 12)
        pts = o + d[ok] * t[ok, None]
        pts = pts + rng.normal(0, noise, pts.shape)
        cams.append((o, pts))
        all_pts.append(pts)
        all_cols.append(palette[wb[ok]])
    return dict(points=np.vstack(all_pts), cams=cams,
                colors=np.vstack(all_cols))


def random_similarity(seed=1, scale=0.7):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return scale, R, rng.normal(0, 1.0, 3)


def transform_scene(S, T):
    s, R, t = T
    f = lambda p: s * (p @ R.T) + t
    return dict(points=f(S["points"]), colors=S["colors"],
                cams=[(f(c[None])[0], f(p)) for c, p in S["cams"]])


def make_pair():
    A = make_scene("A", arc=(-170, -10), seed=0)
    B = make_scene("B", arc=(-150, -40), n_cams=24, seed=1)
    return A, B

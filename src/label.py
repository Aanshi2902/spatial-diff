"""Turn raw added/removed clusters into ADDED / REMOVED / MOVED explanations."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def class_vote(cluster, points, point_classes, names):
    """Majority class among the 3D points inside the cluster's box.

    point_classes: (N,) int array, -1 = no detection. names: list[str].
    """
    if point_classes is None:
        return "object", 0.0
    inside = np.all((points >= cluster["bbox_min"]) & (points <= cluster["bbox_max"]), axis=1)
    votes = point_classes[inside]
    votes = votes[votes >= 0]
    if len(votes) == 0:
        return "object", 0.0
    vals, cnt = np.unique(votes, return_counts=True)
    j = cnt.argmax()
    return names[vals[j]], float(cnt[j] / inside.sum())


def _cost(r, a):
    """Dissimilarity between a removed cluster r and an added cluster a (0 = same)."""
    er, ea = np.sort(r["extent"]), np.sort(a["extent"])
    shape = np.mean(np.abs(er - ea) / np.maximum(np.maximum(er, ea), 1e-6))
    vol = abs(np.log((r["n_voxels"] + 1) / (a["n_voxels"] + 1)))
    col = 0.0
    if r.get("color") is not None and a.get("color") is not None:
        col = float(np.linalg.norm(r["color"] - a["color"]) / np.sqrt(3))
    cls = 0.0 if r["label"] == a["label"] else (0.0 if "object" in (r["label"], a["label"]) else 1.0)
    return shape + 0.5 * vol + 1.0 * col + cls


def explain(added, removed, max_cost=0.6):
    """Match removed<->added clusters; matches are MOVED, leftovers ADDED/REMOVED."""
    changes, used_r, used_a = [], set(), set()
    if added and removed:
        C = np.array([[_cost(r, a) for a in added] for r in removed])
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] <= max_cost:
                r, a = removed[i], added[j]
                disp = a["center"] - r["center"]
                changes.append(dict(kind="MOVED", label=a["label"],
                                    from_center=r["center"], to_center=a["center"],
                                    displacement=disp, distance=float(np.linalg.norm(disp)),
                                    bbox_min=a["bbox_min"], bbox_max=a["bbox_max"],
                                    from_bbox=(r["bbox_min"], r["bbox_max"]),
                                    confidence=float(np.exp(-C[i, j]))))
                used_r.add(i); used_a.add(j)
    for j, a in enumerate(added):
        if j not in used_a:
            changes.append(dict(kind="ADDED", label=a["label"], center=a["center"],
                                bbox_min=a["bbox_min"], bbox_max=a["bbox_max"],
                                confidence=float(min(1.0, a["n_voxels"] / 30))))
    for i, r in enumerate(removed):
        if i not in used_r:
            changes.append(dict(kind="REMOVED", label=r["label"], center=r["center"],
                                bbox_min=r["bbox_min"], bbox_max=r["bbox_max"],
                                confidence=float(min(1.0, r["n_voxels"] / 30))))
    return changes


def sentence(c, unit="units"):
    if c["kind"] == "MOVED":
        return (f"MOVED: {c['label']} moved {c['distance']:.2f} {unit} "
                f"(dx={c['displacement'][0]:+.2f}, dy={c['displacement'][1]:+.2f}, "
                f"dz={c['displacement'][2]:+.2f}).")
    verb = "appeared" if c["kind"] == "ADDED" else "is no longer present"
    x, y, z = c["center"]
    return f"{c['kind']}: {c['label']} {verb} near ({x:.2f}, {y:.2f}, {z:.2f})."

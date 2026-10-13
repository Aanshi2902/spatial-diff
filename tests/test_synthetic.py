"""Run: python tests/test_synthetic.py   (from the repo root)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from synthetic import make_pair, random_similarity, transform_scene, GT
from src import align as al, detect_changes as dc, label as lb

VOX = 0.1


def run_detector(A, B):
    res = dc.detect(A, B, vox=VOX, tol=1, min_voxels=8)
    for c in res["added"] + res["removed"]:
        c["label"] = "object"
    return res, lb.explain(res["added"], res["removed"])


def score(changes):
    ok = 0
    for g in GT:
        match = [c for c in changes if c["kind"] == g["kind"] and
                 np.linalg.norm(np.array(c.get("center", c.get("to_center"))) - g["center"]) < 0.45]
        ok += bool(match)
    return ok, len(changes) - ok


def main():
    A, B = make_pair()
    print(f"scene A: {len(A['points'])} pts, scene B: {len(B['points'])} pts")

    print("\n== 1) Known alignment (isolates the detector) ==")
    res, ch = run_detector(A, B)
    for c in ch:
        print("  ", lb.sentence(c, "m"))
    hit, extra = score(ch)
    print(f"  ground truth found: {hit}/3, extra detections: {extra}, "
          f"unknown-region voxels ignored: {res['n_unknown_voxels']}")

    print("\n== 2) Naive point-distance baseline (ablation) ==")
    nv = dc.naive_diff(A, B, vox=VOX, dist_thresh=0.08)
    print(f"  naive: {len(nv['added'])} added + {len(nv['removed'])} removed clusters "
          f"(truth: 1 added, 1 removed, 1 moved)")

    print("\n== 3) B in a random frame and scale -> align -> detect ==")
    T = random_similarity(seed=3, scale=0.6)
    Bw = transform_scene(B, T)
    Tfix, rmse = al.align(Bw["points"], A["points"], use_open3d=False)
    s, R, t = Tfix
    print(f"  recovered scale {s:.3f} (true {1/T[0]:.3f}), rmse {rmse:.3f} m")
    Bal = transform_scene(Bw, Tfix)
    res3, ch3 = run_detector(A, Bal)
    for c in ch3:
        print("  ", lb.sentence(c, "m"))
    hit3, extra3 = score(ch3)
    print(f"  ground truth found: {hit3}/3, extra detections: {extra3}")


if __name__ == "__main__":
    main()

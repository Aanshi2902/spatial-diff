"""End-to-end: two scene files (or --demo) -> aligned scenes -> 3D changes -> labels -> outputs.

    python run_pipeline.py --demo
    python run_pipeline.py --scene-a data/scene_A.npz --scene-b data/scene_B.npz --out outputs/run1
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from src import align as al, detect_changes as dc, label as lb, visualize as vz


def to_jsonable(c):
    out = {}
    for k, v in c.items():
        if isinstance(v, np.ndarray):
            out[k] = [round(float(x), 4) for x in v]
        elif isinstance(v, tuple):
            out[k] = [[round(float(x), 4) for x in a] for a in v]
        elif isinstance(v, (float, np.floating)):
            out[k] = round(float(v), 4)
        else:
            out[k] = v
    return out


def run(A, B, out_dir, vox_frac=1 / 90, align_scenes=True, unit="units", use_open3d=True):
    os.makedirs(out_dir, exist_ok=True)
    info = {}
    if align_scenes:
        T, rmse = al.align(B["points"], A["points"], use_open3d=use_open3d)
        info.update(scale=float(T[0]), alignment_rmse=float(rmse))
        f = lambda p: al.apply(T, p)
        B = dict(points=f(B["points"]), colors=B.get("colors"),
                 cams=[(f(c[None])[0], f(p)) for c, p in B["cams"]])
        print(f"[pipeline] aligned B->A: scale={T[0]:.3f}, rmse={rmse:.4f}")
    extent = np.linalg.norm(A["points"].max(0) - A["points"].min(0))
    vox = extent * vox_frac
    res = dc.detect(A, B, vox=vox, tol=1, min_voxels=8)
    for c in res["added"] + res["removed"]:
        c["label"] = "object"       # replaced by lb.class_vote(...) once segmentation output is wired in
    changes = lb.explain(res["added"], res["removed"])
    sentences = [lb.sentence(c, unit) for c in changes]
    info.update(voxel_size=float(vox), unknown_voxels_ignored=res["n_unknown_voxels"])

    with open(os.path.join(out_dir, "changes.json"), "w") as fh:
        json.dump(dict(info=info, changes=[to_jsonable(c) for c in changes]), fh, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write("# Detected changes\n\n" + "\n".join(f"- {s}" for s in sentences) + "\n")
    vz.export_html(A["points"], A.get("colors"), B["points"], B.get("colors"), changes,
                   os.path.join(out_dir, "change_map.html"), unit=unit)
    print("\n".join(sentences) or "no changes detected")
    print(f"[pipeline] wrote {out_dir}/changes.json, report.md, change_map.html")
    return changes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use the synthetic room (no videos needed)")
    ap.add_argument("--scene-a"); ap.add_argument("--scene-b")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--no-open3d", action="store_true")
    a = ap.parse_args()
    if a.demo:
        from synthetic import make_pair, random_similarity, transform_scene
        A, B = make_pair()
        B = transform_scene(B, random_similarity(seed=3, scale=0.6))   # B arrives in its own frame/scale
        run(A, B, a.out, unit="m", use_open3d=False)
    else:
        from src.reconstruct import load_scene
        assert a.scene_a and a.scene_b, "give --scene-a and --scene-b, or use --demo"
        run(load_scene(a.scene_a), load_scene(a.scene_b), a.out, use_open3d=not a.no_open3d)

"""Frames -> shared-format scene file (.npz) using VGGT. Run this on a GPU (Colab T4).

!! UNTESTED against the real model in the environment where this was written (no GPU).
!! Check the shapes printed below on your first Colab run; tensor layouts are the most
!! likely place to need a small fix. See notebooks/colab_reconstruct.ipynb.

Output .npz keys (this is the contract the rest of the pipeline relies on):
    points        (N,3) fused, confidence-filtered world points
    colors        (N,3) 0..1
    cam_centers   (F,3) camera centre per frame, same frame as points
    frame_points  (F,M,3) per-frame world points (for free-space ray casting)
    frame_colors  (F,M,3)
"""
import argparse, glob, os
import numpy as np


def reconstruct(frames_dir, out_npz, device="cuda", conf_pct=40, per_frame=6000, seed=0):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    assert paths, f"no frames in {frames_dir}"
    model = VGGT.from_pretrained("facebook/VGGT").to(device).eval()
    images = load_and_preprocess_images(paths).to(device)          # (S,3,H,W)
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                pred = model(images)
        else:
            pred = model(images)

    wp = pred["world_points"].float().cpu().numpy()[0]              # (S,H,W,3)
    conf = pred["world_points_conf"].float().cpu().numpy()[0]       # (S,H,W)
    extr, _ = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extr = extr.float().cpu().numpy()[0]                            # (S,3,4) camera-from-world
    R, t = extr[:, :, :3], extr[:, :, 3]
    centers = -np.einsum("sji,sj->si", R, t)                        # -R^T t
    imgs = images.float().cpu().numpy().transpose(0, 2, 3, 1)       # (S,H,W,3)
    print("[recon] world_points", wp.shape, "conf", conf.shape, "centers", centers.shape)

    rng = np.random.default_rng(seed)
    thr = np.percentile(conf, conf_pct)
    fp, fc, allp, allc = [], [], [], []
    for s in range(len(wp)):
        p, c, cf = wp[s].reshape(-1, 3), imgs[s].reshape(-1, 3), conf[s].reshape(-1)
        ok = (cf >= thr) & np.isfinite(p).all(1)
        p, c = p[ok], c[ok]
        sel = rng.choice(len(p), min(per_frame, len(p)), replace=False)
        fp.append(p[sel]); fc.append(c[sel])
        allp.append(p[::4]); allc.append(c[::4])
    N = min(len(x) for x in fp)
    np.savez_compressed(out_npz, points=np.vstack(allp), colors=np.vstack(allc),
                        cam_centers=centers, frame_points=np.stack([x[:N] for x in fp]),
                        frame_colors=np.stack([x[:N] for x in fc]))
    print(f"[recon] saved {out_npz}")


def load_scene(npz):
    d = np.load(npz)
    cams = [(d["cam_centers"][i], d["frame_points"][i]) for i in range(len(d["cam_centers"]))]
    return dict(points=d["points"], colors=d["colors"], cams=cams)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir"); ap.add_argument("out_npz")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    reconstruct(a.frames_dir, a.out_npz, a.device)

"""Sample frames from a video at a fixed rate, dropping blurry ones."""
import argparse, os
import cv2
import numpy as np


def extract(video, out_dir, fps=1.5, max_frames=50, width=518, blur_keep=0.8):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(src_fps / fps)), 1)
    cands = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
            sharp = cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            cands.append((i, sharp, fr))
        i += 1
    cap.release()
    if len(cands) > max_frames:                       # evenly thin out
        keep = np.linspace(0, len(cands) - 1, max_frames).astype(int)
        cands = [cands[k] for k in keep]
    thr = np.quantile([c[1] for c in cands], 1 - blur_keep) if cands else 0
    paths = []
    for n, (idx, sharp, fr) in enumerate(c for c in cands if c[1] >= thr):
        p = os.path.join(out_dir, f"frame_{n:04d}.jpg")
        cv2.imwrite(p, fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(p)
    print(f"[frames] {video}: {len(paths)} frames -> {out_dir}")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=1.5)
    ap.add_argument("--max-frames", type=int, default=50)
    a = ap.parse_args()
    extract(a.video, a.out_dir, a.fps, a.max_frames)

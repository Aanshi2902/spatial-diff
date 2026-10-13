# spatial-diff

Detect what changed in a room between two phone videos, in 3D.

Two recordings of the same room go in. Out comes a shared 3D reconstruction, a map of what **moved, appeared or disappeared**, plain-language labels, and an interactive 3D viewer.

> **Project status (read this first)**
> | Stage | Status |
> |---|---|
> | Alignment (similarity transform, scale-aware) | Tested on synthetic data |
> | Change detection with free-space reasoning | Tested on synthetic data |
> | Moved / added / removed matching | Tested on synthetic data |
> | HTML viewer, JSON and report outputs | Working |
> | Video -> 3D reconstruction (VGGT, `src/reconstruct.py`) | **Written, not yet run on a GPU or on real video** |
> | Semantic labels (chair, backpack...) from a 2D segmenter lifted to 3D | **Not implemented yet** (`label.class_vote` is ready; segmentation is not wired in) |
> | Results on real recordings | **None yet** |
>
> This table gets updated as real results come in. Numbers below are from a *synthetic* room, not real footage.

## Idea

Comparing two point clouds directly produces false changes: parts of the room one recording never saw look "removed". Here a voxel only counts as **added** when it is occupied in B *and* A's camera rays passed through that space and saw it empty. Space that A never observed is **unknown** and ignored.

```
video A ─┐                                         ┌─ ADDED / REMOVED / MOVED
         ├─ frames ─ VGGT (GPU) ─ scene .npz ─ align B→A ─ free-space voxel diff ─ match ─┤  changes.json
video B ─┘                                         └─ change_map.html, report.md
```

1. `extract_frames.py`: sample about 1.5 fps, drop blurry frames.
2. `reconstruct.py`: poses and per-frame 3D points with VGGT. Needs a GPU, so it runs in Colab.
3. `align.py`: scale-aware alignment of B onto A (global init, then trimmed ICP with scale). Monocular scale is arbitrary, so a rigid transform is not enough.
4. `detect_changes.py`: voxel occupancy plus observed-free-space from camera rays.
5. `label.py`: Hungarian matching of removed and added clusters by shape, size and colour. A match is **MOVED**; leftovers are **ADDED** or **REMOVED**.
6. `visualize.py`: interactive Plotly HTML (green = added, red = removed, yellow = moved, dashed box = old position).

## Run the demo (no videos, no GPU)

```bash
pip install -r requirements.txt
python run_pipeline.py --demo --out outputs/demo
python tests/test_synthetic.py
```
Open `outputs/demo/change_map.html` in a browser. The demo takes about a minute on a CPU.

## Run on real videos

1. Put `video_A.mp4` and `video_B.mp4` in a Drive folder `spatial-diff-data`.
2. Open `notebooks/colab_reconstruct.ipynb` in Colab with a T4 GPU and run all cells.
3. Or, with your own `scene_A.npz` and `scene_B.npz` (format documented in `src/reconstruct.py`):
   `python run_pipeline.py --scene-a data/scene_A.npz --scene-b data/scene_B.npz --out outputs/run1`

## Synthetic results

A ray-cast room with a moved chair, an added backpack and a removed box. Camera path B is shorter than A, so part of the room is seen only in A. Scene B was also given a random rotation, translation and scale (0.6x) before alignment.

| Method | Ground-truth changes found | Extra / false detections |
|---|---|---|
| Naive nearest-neighbour point distance | n/a | 14 clusters for 3 real changes |
| Free-space voxel diff (this repo), known alignment | 3 / 3 | 0 |
| Free-space voxel diff, after recovering random scale + rotation | 3 / 3 | 0 |

Recovered scale 1.668 (true 1.667), alignment RMSE 0.012 m. Voxel size 0.1 m.

## Limitations (known)

- **Synthetic only so far.** Real footage adds reconstruction noise, lighting changes, reflective and thin objects, and imperfect alignment. Expect more tuning than the synthetic numbers suggest.
- **Voxel size matters.** At a coarser voxel size (about 0.15 m) the small synthetic backpack fell below the minimum cluster size and was missed. The default is scene extent / 90; `min_voxels` and `tol` may need adjusting per scene.
- **Matching thresholds are heuristics.** With similar-looking, similarly coloured objects, a removed object and an added one can be wrongly reported as one MOVED object. The `max_cost` threshold (0.6) was tuned on one synthetic scene.
- **Alignment fallback is weaker than the main path.** Without Open3D the global step tries 24 PCA-axis orientations. Open3D's FPFH + RANSAC is the intended global method but has not been exercised here.
- **Scale is relative** unless a known-size object is used to calibrate. Distances are reported in scene units (metres in the synthetic demo).
- **Labels are generic ("object")** until a segmenter is wired into `label.class_vote`.
- **Free-space reasoning needs camera centres** from the reconstruction. If poses are wrong, free space is wrong.

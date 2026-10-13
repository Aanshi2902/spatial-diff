"""Align scene B onto scene A with a similarity transform (scale, rotation, translation).

Monocular reconstructions have arbitrary scale, so a rigid transform is not enough.
Stage 1 (global): Open3D FPFH+RANSAC if installed, otherwise PCA-axis candidates.
Stage 2 (refine): trimmed point-to-point ICP with scale, pure numpy/scipy.
"""
import itertools
import numpy as np
from scipy.spatial import cKDTree


def umeyama(src, dst, with_scale=True):
    """Least-squares s,R,t such that dst ~= s*R@src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (xs ** 2).sum() / len(src)
    s = (S * np.diag(D)).sum() / var if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply(T, p):
    s, R, t = T
    return s * (p @ R.T) + t


def compose(T2, T1):
    """T2 after T1."""
    s1, R1, t1 = T1
    s2, R2, t2 = T2
    return s2 * s1, R2 @ R1, s2 * (R2 @ t1) + t2


def icp(src, dst, T0, iters=60, trim=0.7, max_pts=20000, seed=0, tree=None):
    rng = np.random.default_rng(seed)
    if len(src) > max_pts:
        src = src[rng.choice(len(src), max_pts, replace=False)]
    if tree is None:
        tree = cKDTree(dst)
    T = T0
    for _ in range(iters):
        moved = apply(T, src)
        d, j = tree.query(moved)
        keep = d <= np.quantile(d, trim)
        T = compose(umeyama(moved[keep], dst[j[keep]]), T)
    moved = apply(T, src)
    d, _ = tree.query(moved)
    return T, float(np.sqrt(np.mean(np.sort(d)[: int(trim * len(d))] ** 2)))


def _normalize(p):
    c = p.mean(0)
    s = np.sqrt(((p - c) ** 2).sum(1).mean())
    return c, s


def _pca_candidates(srcn, dstn):
    """24 proper axis permutations/sign flips in PCA frames."""
    def frame(p):
        _, _, Vt = np.linalg.svd(p - p.mean(0), full_matrices=False)
        if np.linalg.det(Vt) < 0:
            Vt[2] *= -1
        return Vt
    Fs, Fd = frame(srcn), frame(dstn)
    cands = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for r, (c, sg) in enumerate(zip(perm, signs)):
                M[r, c] = sg
            if np.linalg.det(M) > 0:
                cands.append(Fd.T @ M @ Fs)
    return cands


def _open3d_global(src, dst, voxel):
    import open3d as o3d
    def prep(p):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        pc = pc.voxel_down_sample(voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
        f = o3d.pipelines.registration.compute_fpfh_feature(
            pc, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
        return pc, f
    (ps, fs), (pd, fd) = prep(src), prep(dst)
    r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        ps, pd, fs, fd, True, voxel * 2,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 2)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999))
    M = np.asarray(r.transformation)
    A = M[:3, :3]
    s = np.cbrt(abs(np.linalg.det(A)))
    return s, A / s, M[:3, 3]


def align(src, dst, use_open3d=True, verbose=True):
    """Return (T, rmse) mapping src (scene B) into dst (scene A) coordinates."""
    cs, ss = _normalize(src)
    cd, sd = _normalize(dst)
    srcn, dstn = (src - cs) / ss, (dst - cd) / sd
    to_n_src = (1 / ss, np.eye(3), -cs / ss)
    from_n_dst = (sd, np.eye(3), cd)

    best = None
    if use_open3d:
        try:
            T = _open3d_global(srcn, dstn, voxel=0.04)
            T, rmse = icp(srcn, dstn, T)
            best = (rmse, T)
            if verbose:
                print(f"[align] open3d global + ICP rmse={rmse:.4f} (normalized units)")
        except Exception as e:  # open3d missing or failed
            if verbose:
                print(f"[align] open3d unavailable ({type(e).__name__}); using PCA fallback")
    if best is None:
        rng = np.random.default_rng(0)
        small = dstn[rng.choice(len(dstn), min(len(dstn), 8000), replace=False)]
        small_tree = cKDTree(small)
        scored = []
        for R in _pca_candidates(srcn, dstn):      # cheap coarse scoring of 24 starts
            T, rmse = icp(srcn, small, (1.0, R, np.zeros(3)), iters=25, max_pts=2000,
                          tree=small_tree)
            scored.append((rmse, T))
        scored.sort(key=lambda x: x[0])
        full_tree = cKDTree(dstn)
        best = None
        for _, T0 in scored[:3]:                   # refine only the best three
            T, rmse = icp(srcn, dstn, T0, iters=60, tree=full_tree)
            if best is None or rmse < best[0]:
                best = (rmse, T)
        if verbose:
            print(f"[align] PCA candidates + ICP rmse={rmse:.4f} (normalized units)")
    rmse, T = best
    full = compose(from_n_dst, compose(T, to_n_src))
    return full, rmse * sd

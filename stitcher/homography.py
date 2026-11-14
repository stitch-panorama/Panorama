import numpy as np

def normalize(pts):
    # shift points to zero mean, scale so average distance is sqrt(2)
    c = pts.mean(axis=0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.array([[s, 0, -s * c[0]],
                  [0, s, -s * c[1]],
                  [0, 0, 1]])
    pts_h = np.c_[pts, np.ones(len(pts))]
    return (T @ pts_h.T).T[:, :2], T

def estimate_homography(src, dst):
    """src, dst: (N, 2) arrays with N >= 4. Returns 3x3 H mapping src -> dst."""
    src_n, T1 = normalize(src)
    dst_n, T2 = normalize(dst)
    A = []
    for (x, y), (xp, yp) in zip(src_n, dst_n):
        A.append([-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp])
        A.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
    _, _, Vt = np.linalg.svd(np.array(A))   # keep full_matrices=True (the default)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1          # undo the normalization
    return H / H[2, 2]

def apply_homography(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def ransac_homography(src, dst, thresh=3.0, p=0.99, max_iters=5000, seed=0):
    """Returns (H, inlier_mask, iterations_used), or (None, None, iterations_used)."""
    rng = np.random.default_rng(seed)
    n = len(src)
    best_mask, best_count = None, 0
    iters_needed = max_iters
    i = 0
    while i < min(iters_needed, max_iters):
        i += 1
        idx = rng.choice(n, 4, replace=False)
        try:
            H = estimate_homography(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(H)):
            continue
        with np.errstate(all="ignore"):
            err = np.linalg.norm(apply_homography(H, src) - dst, axis=1)
        mask = err < thresh
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
            w = count / n
            iters_needed = 0 if w >= 1 else np.log(1 - p) / np.log(1 - w ** 4)
    if best_mask is None or best_count < 4:
        return None, None, i
    # refit on all inliers of the best model
    H = estimate_homography(src[best_mask], dst[best_mask])
    return H, best_mask, i
#!/usr/bin/env python3
"""
FAISS GPU K-means + save per-iteration objective curve to a PNG.

Run (single GPU recommended):
  CUDA_VISIBLE_DEVICES=0 python test_kmeans.py --k 8192 --niter 30 --out_png obj.png

Use your data:
  CUDA_VISIBLE_DEVICES=0 python test_kmeans.py --input X.npy --k 8192 --niter 30 --out_png obj.png

Notes:
- Input shape: (n, 6, 3) or (n, 18)
- FAISS expects float32 contiguous arrays
- Per-iteration objective values are in kmeans.obj (total squared error for k-means).  [oai_citation:1‡GitHub](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks%3A-clustering%2C-PCA%2C-quantization?utm_source=chatgpt.com)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import faiss  # import early (helps avoid some env-specific segfaults)


def load_X(path: str) -> np.ndarray:
    """Load .npy (or .npz with key 'X') into float32 contiguous array of shape (n, 18)."""
    if path.endswith(".npz"):
        z = np.load(path)
        if "X" not in z:
            raise ValueError(f"{path} is .npz but missing key 'X'. Keys={list(z.keys())}")
        X = z["X"]
    else:
        X = np.load(path)

    if X.ndim == 3:
        if X.shape[1:] != (6, 3):
            raise ValueError(f"Expected (n,6,3) if 3D, got {X.shape}")
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 2:
        if X.shape[1] != 18:
            raise ValueError(f"Expected (n,18) if 2D, got {X.shape}")
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape={X.shape}")

    return np.asarray(X, dtype=np.float32, order="C")


def standardize_inplace(X: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize per feature in-place. Returns (mean, std)."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    X -= mean
    X /= std
    return mean.astype(np.float32), std.astype(np.float32)


def run_faiss_kmeans_gpu(
    X: np.ndarray,
    k: int,
    niter: int,
    nredo: int,
    seed: int,
    verbose: bool,
    min_points_per_centroid: int,
    max_points_per_centroid: int,
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """Return (obj_per_iter, centroids, labels)."""
    ngpu = faiss.get_num_gpus()
    print(f"[faiss] version={getattr(faiss, '__version__', 'unknown')} ngpu={ngpu}")
    if ngpu <= 0:
        raise RuntimeError("FAISS reports 0 GPUs. Fix faiss-gpu/CUDA install first.")

    n, d = X.shape
    print(f"[data] X shape={X.shape} k={k} niter={niter} nredo={nredo}")

    # IMPORTANT:
    # max_points_per_centroid is NOT "0 means no cap".
    # Default in FAISS is 256; it limits/subsamples the training set size.  [oai_citation:2‡faiss.ai](https://faiss.ai/cpp_api/file/Clustering_8h.html?utm_source=chatgpt.com)
    if max_points_per_centroid <= 0:
        raise ValueError("--max_points_per_centroid must be a positive integer (try 256 or larger).")

    kmeans = faiss.Kmeans(
        d=d,
        k=k,
        niter=niter,
        nredo=nredo,
        seed=seed,
        gpu=True,  # documented path: uses available GPUs  [oai_citation:3‡GitHub](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks%3A-clustering%2C-PCA%2C-quantization?utm_source=chatgpt.com)
        verbose=verbose,
        min_points_per_centroid=min_points_per_centroid,
        max_points_per_centroid=max_points_per_centroid,
    )

    kmeans.train(X)

    obj = [float(v) for v in kmeans.obj]  # per-iter objective history  [oai_citation:4‡GitHub](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks%3A-clustering%2C-PCA%2C-quantization?utm_source=chatgpt.com)
    centroids = kmeans.centroids.reshape(k, d).astype(np.float32).reshape(k, 6, 3)

    # labels via nearest centroid
    _, I = kmeans.index.search(X, 1)
    labels = I.reshape(-1).astype(np.int32)

    return obj, centroids, labels


def save_obj_png(obj: List[float], out_path: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.figure()
    plt.plot(range(1, len(obj) + 1), obj, marker="o")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Objective (SSE / inertia)")
    plt.title("FAISS k-means objective per iteration (L2; lower is better)")
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def save_centroid_trajectories_png(centroids: np.ndarray, out_path: str, seed: int = 0) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # centroids: (k, 6, 3) where last dim is (x, y, yaw)
    xy = centroids[:, :, :2]  # (k, 6, 2)

    rng = np.random.default_rng(seed)
    colors = rng.random((centroids.shape[0], 3))  # random RGB per trajectory

    plt.figure()
    for i in range(xy.shape[0]):
        plt.plot(xy[i, :, 0], xy[i, :, 1], color=colors[i], linewidth=1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Centroid trajectories (x,y)")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def save_centroid_yaw_arrows_png(
    centroids: np.ndarray,
    out_path: str,
    n_traj: int = 12,
    arrow_stride: int = 1,
    arrow_scale: float = 0.5,
    seed: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # centroids: (k, 6, 3) where last dim is (x, y, yaw)
    k = centroids.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(k, size=min(n_traj, k), replace=False)

    plt.figure()
    for j in idx:
        traj = centroids[j]  # (6, 3)
        x = traj[:, 0]
        y = traj[:, 1]
        yaw = traj[:, 2]

        color = rng.random(3)
        plt.plot(x, y, color=color, linewidth=2)

        dx = np.cos(yaw) * arrow_scale
        dy = np.sin(yaw) * arrow_scale

        pts = np.arange(0, len(x), max(1, arrow_stride))
        plt.quiver(
            x[pts],
            y[pts],
            dx[pts],
            dy[pts],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.004,
            headwidth=3.5,
            headlength=5,
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sample centroid trajectories with yaw arrows")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="", help="Path to .npy or .npz (with key 'X').")
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--niter", type=int, default=30)
    p.add_argument("--nredo", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--out_png", type=str, default="faiss_kmeans_objective.png")
    p.add_argument("--out_traj_png", type=str, default="faiss_kmeans_centroid_trajectories.png")
    p.add_argument("--out_yaw_png", type=str, default="faiss_kmeans_centroid_yaw_arrows.png")
    p.add_argument("--yaw_plot_n", type=int, default=12)
    p.add_argument("--yaw_arrow_stride", type=int, default=1)
    p.add_argument("--yaw_arrow_scale", type=float, default=0.5)
    p.add_argument("--save_centroids", type=str, default="")
    p.add_argument("--save_labels", type=str, default="")
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--min_points_per_centroid", type=int, default=10)

    # Keep FAISS default behavior unless you have a reason to change it.
    # Default in FAISS clustering parameters is 256.  [oai_citation:5‡faiss.ai](https://faiss.ai/cpp_api/file/Clustering_8h.html?utm_source=chatgpt.com)
    p.add_argument("--max_points_per_centroid", type=int, default=256,
                   help="Positive int. Limits training set via subsampling to k*max_points_per_centroid.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        X = load_X(args.input)
    else:
        X = np.random.randn(200000, 18).astype(np.float32)
        print("[warn] No --input provided. Using synthetic random data:", X.shape)

    if args.standardize:
        print("[info] Standardizing features...")
        standardize_inplace(X)

    obj, centroids, labels = run_faiss_kmeans_gpu(
        X=X,
        k=args.k,
        niter=args.niter,
        nredo=args.nredo,
        seed=args.seed,
        verbose=args.verbose,
        min_points_per_centroid=args.min_points_per_centroid,
        max_points_per_centroid=args.max_points_per_centroid,
    )

    print(f"[done] iters={len(obj)} final_objective={obj[-1]:.6g}")
    save_obj_png(obj, args.out_png)
    save_centroid_trajectories_png(centroids, args.out_traj_png, seed=args.seed)
    save_centroid_yaw_arrows_png(
        centroids,
        args.out_yaw_png,
        n_traj=args.yaw_plot_n,
        arrow_stride=args.yaw_arrow_stride,
        arrow_scale=args.yaw_arrow_scale,
        seed=args.seed,
    )

    if args.save_centroids:
        os.makedirs(os.path.dirname(args.save_centroids) or ".", exist_ok=True)
        np.save(args.save_centroids, centroids)
        print(f"[saved] {args.save_centroids} (centroids {centroids.shape})")

    if args.save_labels:
        os.makedirs(os.path.dirname(args.save_labels) or ".", exist_ok=True)
        np.save(args.save_labels, labels)
        print(f"[saved] {args.save_labels} (labels {labels.shape})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

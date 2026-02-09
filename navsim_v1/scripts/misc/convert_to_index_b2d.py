import concurrent.futures
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


def convert_one(metric_cache_path: str):
    new_metric_path = Path(
        metric_cache_path.replace("b2d_anchors_scores", "b2d_anchors_scores_index")
    )
    new_metric_path.parent.mkdir(parents=True, exist_ok=True)

    scores = np.load(metric_cache_path)

    # find indices where last column >= 0.95
    idx = np.where(scores[:, -1] >= 0.95)[0]

    # if more than 256, randomly sample 256 indices
    if len(idx) > 256:
        idx = np.random.choice(idx, size=256, replace=False)

    # save indices to new path
    np.save(new_metric_path, idx)


if __name__ == "__main__":
    scores_path = "/scratch/linhan/b2d_anchors_scores_8192/"
    max_workers = 32

    file_names = sorted(str(p) for p in Path(scores_path).rglob("*.npy"))

    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(
            executor.map(convert_one, file_names, chunksize=1), total=len(file_names)
        ):
            pass
    print(
        f"Completed pruning {len(file_names)} metric caches "
        f"in {time.time() - start:.1f}s"
    )

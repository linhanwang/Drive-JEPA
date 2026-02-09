import time
from pathlib import Path
import numpy as np
import csv
import concurrent.futures
from tqdm import tqdm
import argparse


def convert_one(metric_cache_path: str):
    new_metric_path = Path(metric_cache_path.replace('anchors_scores_v1', 'anchors_scores_index_v1'))
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/linhan/yinlin/projects/navsim_workspace/exp/anchors_scores_v1/metadata/anchors_scores_v1_metadata_node_0.csv"
    )
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args()

    csv_path = args.csv_path

    file_names = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        file_names = [str(row["file_name"]) for row in reader]

    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for _ in tqdm(executor.map(convert_one, file_names, chunksize=1), total=len(file_names)):
            pass
    print(
        f"Completed pruning {len(file_names)} metric caches "
        f"in {time.time() - start:.1f}s"
    )

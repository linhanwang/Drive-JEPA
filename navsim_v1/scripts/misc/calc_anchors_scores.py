import time
from pathlib import Path
import numpy as np
import csv
import concurrent.futures
from tqdm import tqdm
import argparse
from navsim.agents.pad_vjepa.score_module.compute_navsim_score import get_sub_score


def convert_one(metric_cache_path: str):
    new_metric_path = Path(metric_cache_path.replace('train_ipad_metric_cache', 'anchors_scores_v1'))
    new_metric_path.parent.mkdir(parents=True, exist_ok=True)

    poses = np.load("./data/8192.npy")
    poses = poses[:, 4::5]

    scores, _, _, _ = get_sub_score(metric_cache_path, poses, False)
    np.save(new_metric_path, scores)


def get_split(items, num_splits: int, split_idx: int):
    n = len(items)
    base = n // num_splits
    rem = n % num_splits
    if split_idx < rem:
        start = split_idx * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (split_idx - rem) * base
        end = start + base
    return items[start:end]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/linhan/yinlin/projects/navsim_workspace/exp/train_ipad_metric_cache/metadata/train_ipad_metric_cache_metadata_node_0.csv",
    )
    parser.add_argument("--num_splits", type=int, default=10)
    parser.add_argument("--split_idx", type=int, required=True)
    parser.add_argument("--max_workers", type=int, default=32)
    args = parser.parse_args()

    csv_path = args.csv_path

    file_names = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        file_names = [str(row["file_name"]) for row in reader]

    if not (0 <= args.split_idx < args.num_splits):
        raise ValueError(f"split_idx must be in [0, {args.num_splits - 1}], got {args.split_idx}")

    split_file_names = get_split(file_names, args.num_splits, args.split_idx)

    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for _ in tqdm(executor.map(convert_one, split_file_names, chunksize=1), total=len(split_file_names)):
            pass
    print(
        f"Completed pruning {len(split_file_names)} metric caches "
        f"for split {args.split_idx}/{args.num_splits} in {time.time() - start:.1f}s"
    )

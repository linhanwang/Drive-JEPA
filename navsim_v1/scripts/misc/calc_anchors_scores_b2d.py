import gzip
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from tqdm import tqdm

from navsim.agents.pad.score_module.compute_b2d_score import (
    b2d_before_score,
    compute_corners_torch,
    get_scores,
)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(
    path: Path, data_dict: Dict[str, torch.Tensor]
) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        log_names: Optional[List[str]] = None,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)

        if log_names is not None:
            self.log_names = [
                Path(log_name)
                for log_name in log_names
                if (self._cache_path / log_name).is_dir()
            ]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]

        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,
            log_names=self.log_names,
        )
        self.tokens = list(self._valid_cache_paths.keys())

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(self.tokens[idx])

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                # found_caches: List[bool] = []
                # for builder in feature_builders + target_builders:
                #     data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                #     found_caches.append(data_dict_path.is_file())
                # if all(found_caches):
                valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _load_scene_with_token(
        self, token: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for name in ["pad_feature"]:
            data_dict_path = token_path / (name + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for name in ["pad_target"]:
            data_dict_path = token_path / (name + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)


def main():
    dataset = CacheOnlyDataset("/scratch/linhan/bench2drive_cache", ["train"])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=16)

    train_metric_cache_paths = load_feature_target_from_pickle(
        os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/train_fut_boxes.gz"
    )

    map_file = os.getenv("NAVSIM_EXP_ROOT") + "/map.pkl"

    with open(map_file, "rb") as f:
        map_infos = pickle.load(f)

    # arr: (4096, 6, 2) -> (num_trajectories, num_waypoints, xy)
    arr = np.load("../iPad/data/b2d_8192.npy")

    arr_with_yaw = torch.from_numpy(arr).float()  # (N, T, 3)

    output_dir = Path("/scratch/linhan/b2d_anchors_scores_8192/")
    output_dir.mkdir(exist_ok=True)

    for features, targets in tqdm(dataloader, desc="Iterating batches"):
        bs = len(targets["token"])
        proposals = arr_with_yaw.unsqueeze(0).repeat(bs, 1, 1, 1).cuda()

        targets = {
            k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in targets.items()
        }

        data_points = b2d_before_score(
            map_infos, proposals, targets, train_metric_cache_paths
        )
        all_res = worker_map(worker, get_scores, data_points)
        for token, res in zip(targets["token"], all_res):
            np.save(output_dir / token, res[0])

if __name__ == "__main__":
    main()

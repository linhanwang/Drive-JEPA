import gzip
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


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
    dataset = CacheOnlyDataset("/scratch/linhan/bench2drive_cache", ["train", "val"])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    n = len(dataset)

    trajectories_all: Optional[np.ndarray] = None
    write_pos = 0

    for features, targets in tqdm(dataloader, desc="Iterating batches"):
        batch_np = targets["trajectory"].detach().cpu().numpy()

        if trajectories_all is None:
            trajectories_all = np.empty((n,) + batch_np.shape[1:], dtype=batch_np.dtype)

        bsz = batch_np.shape[0]
        trajectories_all[write_pos : write_pos + bsz] = batch_np
        write_pos += bsz

    trajectories_all = trajectories_all[:write_pos]
    np.save("all_trajectories.npy", trajectories_all)


if __name__ == "__main__":
    main()

from typing import Any, Dict, List, Optional, Union

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drive_jepa_perception_free_b.drive_jepa_features import DriveJEPAFeatureBuilder, DriveJEPAFeatureDIBuilder 
from navsim.agents.drive_jepa_perception_free_b.drive_jepa_model import DriveJEPAModel
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """Input target builder of LAW."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        """

        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}


class DriveJEPAAgent(AbstractAgent):
    """DriveJEPA agent interface."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        pretrain_pt_path: str,
        image_architecture: str = "vit_large",
        tf_d_model: int = 256,
        tf_d_ffn: int = 1024,
        tf_num_layers: int = 3,
        tf_num_head: int = 8,
        tf_dropout: float = 0.0,
        num_keyval: int = 8 * 32 + 1,
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        front_only: bool = False,
        freeze_encoder: bool = True,
        double_image: bool = False,
    ):
        """
        Initializes the agent interface for EgoStatusMLP.
        :param trajectory_sampling: trajectory sampling specification.
        :param hidden_layer_dim: dimensionality of hidden layer.
        :param lr: learning rate during training.
        :param checkpoint_path: optional checkpoint path as string, defaults to None
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path

        self._front_only = front_only
        self._double_image = double_image
        self._lr = lr
        self._model = DriveJEPAModel(
            trajectory_sampling,
            pretrain_pt_path,
            image_architecture,
            tf_d_model,
            tf_d_ffn,
            tf_num_layers,
            tf_num_head,
            tf_dropout,
            num_keyval,
            front_only,
            freeze_encoder,
            double_image,
        )

    def name(self) -> str:
        """Inherited, see superclass."""
        return 'drive_jepa_perception_free_agent' 

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        ilist = [2, 3] if self._double_image else [3]
        if self._front_only:
            return SensorConfig.build_front_only_sensors(include=ilist)
        else:
            return SensorConfig.build_all_sensors(include=ilist)

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        if self._double_image:
            return [DriveJEPAFeatureDIBuilder(self._front_only)]
        else:
            return [DriveJEPAFeatureBuilder(self._front_only)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        if self._double_image:
            cam_f_2 = features['camera_feature_2']
            cam_f_1 = features['camera_feature_1']
            cam_feature = torch.cat([cam_f_2[:, :, None], cam_f_1[:, :, None]], dim=2)
            return self._model(cam_feature, features["status_feature"])
        else:
            return self._model(features["camera_feature"], features["status_feature"])

    def compute_loss(
        self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        
        pred, gt = predictions["trajectory"], targets["trajectory"]
        return l1_length_normalized_loss(pred, gt, alpha=5.0)
        # l1_loss = l1_length_normalized_loss(pred, gt, alpha=5.0)
        # lateral_loss = torch.mean(lateral_distance_pred_to_gt(pred[:, :, :2], gt[:, :, :2]))
        # loss = l1_loss + 0.5 * lateral_loss
        # return {'loss': loss, 'l1_loss': l1_loss, 'lateral_loss': lateral_loss}
        # return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._model.parameters(), lr=self._lr)


def l1_length_normalized_loss(pred, gt, alpha=1.0, eps=1e-6):
    """
    pred, gt: [B, N, 3]  (x, y, yaw)
    Normalizes L1 loss by GT trajectory length.
    """
    # Mean L1 per sample
    per_sample_l1 = (pred - gt).abs().mean(dim=[1, 2])  # [B]

    # Compute GT arc length in XY
    dxy = gt[:, 1:, :2] - gt[:, :-1, :2]                # [B, N-1, 2]
    arc_len = torch.linalg.norm(dxy, dim=-1).sum(dim=1) # [B]
    
    # Compute weights: inverse of (1 + length)
    w = 1.0 / (alpha + arc_len)

    # Normalize weights so avg weight ≈ 1
    w = w * (w.numel() / (w.sum() + eps))

    # Weighted mean loss
    loss = (w * per_sample_l1).mean()
    return loss

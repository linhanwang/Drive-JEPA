from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from navsim.common.dataclasses import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


class DriveJEPAFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for DriveJEPA."""

    def __init__(self, front_only: bool = False):
        """
        Initializes feature builder.
        """
        self.front_only = front_only

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "drive_jepa_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features: Dict[str, torch.Tensor] = {}

        features["camera_feature"] = self._get_camera_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput.
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        cameras = agent_input.cameras[-1]

        if self.front_only:
            # Crop to ensure 2:1 aspect ratio
            f0 = cameras.cam_f0.image[28:-28]

            resized_image = cv2.resize(f0, (512, 256))
        else:
            # Crop to ensure 4:1 aspect ratio
            l0 = cameras.cam_l0.image[28:-28, 416:-416]
            f0 = cameras.cam_f0.image[28:-28]
            r0 = cameras.cam_r0.image[28:-28, 416:-416]

            # stitch l0, f0, r0 images
            stitched_image = np.concatenate([l0, f0, r0], axis=1)
            resized_image = cv2.resize(stitched_image, (1024, 256))

        tensor_image = transforms.ToTensor()(resized_image)
        return tensor_image


class DriveJEPAFeatureDIBuilder(DriveJEPAFeatureBuilder):
    """Input feature builder for DriveJEPA (double-image)."""

    def __init__(self, front_only: bool = False):
        """
        Initializes feature builder.
        """
        self.front_only = front_only

    def _get_camera_feature(self, agent_input: AgentInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract stitched camera from AgentInput.
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        cameras = agent_input.cameras

        if self.front_only:
            # Current frame
            f0 = cameras[-1].cam_f0.image[28:-28]
            resized_image_1 = cv2.resize(f0, (512, 256))

            # Previous frame
            f0 = cameras[-2].cam_f0.image[28:-28]
            resized_image_2 = cv2.resize(f0, (512, 256))
        else:
            # Current frame
            l0 = cameras[-1].cam_l0.image[28:-28, 416:-416]
            f0 = cameras[-1].cam_f0.image[28:-28]
            r0 = cameras[-1].cam_r0.image[28:-28, 416:-416]
            stitched_image = np.concatenate([l0, f0, r0], axis=1)
            resized_image_1 = cv2.resize(stitched_image, (1024, 256))

            # Previous frame
            l0 = cameras[-2].cam_l0.image[28:-28, 416:-416]
            f0 = cameras[-2].cam_f0.image[28:-28]
            r0 = cameras[-2].cam_r0.image[28:-28, 416:-416]
            stitched_image = np.concatenate([l0, f0, r0], axis=1)
            resized_image_2 = cv2.resize(stitched_image, (1024, 256))

        tensor_image_1 = transforms.ToTensor()(resized_image_1)
        tensor_image_2 = transforms.ToTensor()(resized_image_2)

        return tensor_image_1, tensor_image_2

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features: Dict[str, torch.Tensor] = {}

        tensor_image_1, tensor_image_2 = self._get_camera_feature(agent_input)
        features["camera_feature_1"] = tensor_image_1
        features["camera_feature_2"] = tensor_image_2
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        return features

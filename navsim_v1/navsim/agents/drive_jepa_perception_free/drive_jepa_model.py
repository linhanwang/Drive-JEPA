from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import yaml
from einops import rearrange
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.enums import StateSE2Index
from vjepa2.evals.image_classification_frozen.modelcustom.vit_encoder import init_module

MODEL_DICT = {
    "vit_large": {"filename": "vitl.pt", "dim": 1024},
    "vit_huge": {"filename": "vith.pt", "dim": 1280},
    "vit_giant_xformers": {"filename": "vitg.pt", "dim": 1408},
}


class DriveJEPAModel(nn.Module):
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
        front_only: bool = True,
        freeze_encoder: bool = True,
        double_image: bool = False,
    ):
        super().__init__()
        self._double_image = double_image
        fname = "./vjepa2/configs/eval/vitl/in1k.yaml"
        with open(fname, "r") as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)

            resolution = (256, 512) if front_only else (256, 1024)
            model_kwargs = params["model_kwargs"]
            wrapper_kwargs = model_kwargs["wrapper_kwargs"]
            wrapper_kwargs["img_as_video_nframes"] = 2
            model_kwargs = model_kwargs["pretrain_kwargs"]
            model_kwargs["encoder"]["model_name"] = image_architecture
            self.image_encoder = init_module(
                resolution, pretrain_pt_path, model_kwargs, wrapper_kwargs, register_prehook=not self._double_image
            )
            self.freeze_encoder = freeze_encoder
            if self.freeze_encoder:
                # Freeze encoder
                self.image_encoder.eval()
                for p in self.image_encoder.parameters():
                    p.requires_grad = False
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.image_fc = nn.Linear(MODEL_DICT[image_architecture]["dim"], tf_d_model)
        self._status_encoding = nn.Linear(4 + 2 + 2, tf_d_model)

        num_poses = trajectory_sampling.num_poses

        self._keyval_embedding = nn.Embedding(num_keyval, tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(num_poses, tf_d_model)

        self._transformer = nn.Transformer(
            d_model=tf_d_model,
            nhead=tf_num_head,
            num_encoder_layers=tf_num_layers,
            num_decoder_layers=tf_num_layers,
            dim_feedforward=tf_d_ffn,
            dropout=tf_dropout,
            batch_first=True,
        )
        self._trajectory_head = DriveJEPATrajectoryHead(num_poses, tf_d_ffn, tf_d_model)
        self.transform = self.make_transform()

    def make_transform(self):
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([normalize])

    def forward(self, camera_feature, status_feature) -> torch.Tensor:
        batch_size = camera_feature.shape[0]
        H, W = camera_feature.shape[-2:]
        
        # Normalization
        camera_feature = rearrange(camera_feature, 'B C S H W -> (B S) C H W')
        camera_feature = self.transform(camera_feature)
        camera_feature = rearrange(camera_feature, '(B S) C H W -> B C S H W', S=2) # Assume using double image

        if self.freeze_encoder:
            with torch.no_grad():
                img_feat = self.image_encoder(camera_feature)
        else:
            img_feat = self.image_encoder(camera_feature)
        img_feat = rearrange(img_feat, "B (H W) D -> B D H W", H=H // 16, W=W // 16)
        img_feat = self.avg_pool(img_feat)
        img_feat = img_feat.flatten(-2, -1).permute(0, 2, 1)
        img_feat = self.image_fc(img_feat.clone())  # 512 -> 256

        status_encoding = self._status_encoding(status_feature)

        keyval = torch.cat([img_feat, status_encoding[:, None]], dim=1)
        keyval = keyval.clone() + self._keyval_embedding.weight[None, ...]

        keyval_final = keyval

        # traj decoder
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._transformer(src=keyval_final, tgt=query)
        trajectory = self._trajectory_head(query_out)
        return trajectory


class DriveJEPATrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(DriveJEPATrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}

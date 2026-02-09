import torch
import torch.nn as nn
from ..bevformer.transformer_decoder import MyTransformeDecoder

class MapHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model=config.tf_d_model

        self.map_trans=MyTransformeDecoder(config,8*8,self.d_model,trajenc=False )#build position

        channel = 64
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.upsample2 = nn.Upsample(
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )

        self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)  # channel=64
        self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

        # lateral
        self.c5_conv = nn.Conv2d(
            self.d_model, channel, (1, 1)
        )  # 512

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

    def top_down(self, x):

        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))

        return p3

    def forward(self, keyval):

        map_bev=self.map_trans(None,keyval).reshape(-1,8,8,self.d_model).permute(0,3,1,2)

        bev_upscale=self.top_down(map_bev)

        out=self._bev_semantic_head(bev_upscale)

        return out
# if bev_w == 64:
#     self.conv = nn.Conv2d(
#         d_model, channel, (1, 1)
#     )  # 512
# else:

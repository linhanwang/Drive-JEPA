import torch
import torch.nn as nn
from einops import rearrange
import yaml
from vjepa2.evals.image_classification_frozen.modelcustom.vit_encoder import init_module
import os

from .grid_mask import GridMask
from ..drive_jepa_config import DriveJEPAConfig

class ImgEncoder(nn.Module):
    def __init__(self, config: DriveJEPAConfig, num_feature_levels=2):
        super().__init__()
        self.embed_dims = config.tf_d_model
        self.num_feature_levels = num_feature_levels
        num_cams = 1

        self.num_cams = num_cams
        self.use_lidar=False

        self.grid_mask = GridMask( True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = True 
        
        image_architecture = "vit_large"
        pretrain_pt_path = f"{os.getenv('NAVSIM_EXP_ROOT')}/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt"
        fname = "./vjepa2/configs/eval/vitl/in1k.yaml"
        with open(fname, "r") as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)

            resolution = (256, 512)
            model_kwargs = params["model_kwargs"]
            wrapper_kwargs = model_kwargs["wrapper_kwargs"]
            wrapper_kwargs["img_as_video_nframes"] = 2
            model_kwargs = model_kwargs["pretrain_kwargs"]
            model_kwargs["encoder"]["model_name"] = image_architecture
            self.img_backbone = init_module(resolution, pretrain_pt_path, model_kwargs, wrapper_kwargs, register_prehook=False)

        self.projector = nn.Linear(1024, self.embed_dims)

    def forward(self,img,len_queue=None,**kwargs):

        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img = rearrange(img, '(B N) C H W -> B C N H W', B=B)
        img_feat = self.img_backbone(img)#7,12
        img_feat = self.projector(img_feat)
        img_feat = rearrange(img_feat, 'B (H W) C -> B C H W', H=16, W=32)

        BN, C, H, W = img_feat.size()
        feat = img_feat.view(B, int(BN / B), C, H, W)

        bs, num_cam, c, h, w = feat.shape#1,6,256,12,20
        spatial_shape = (h, w)
        feat = feat.flatten(3).permute(1, 0, 3, 2)#6,1,240,256
        #feat = feat +lidar2img_embed[:,:,None]

        spatial_shape = torch.as_tensor(
            [spatial_shape], dtype=torch.long, device=feat.device)
        level_start_index = torch.cat((spatial_shape.new_zeros(
            (1,)), spatial_shape.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        return feat_flatten, spatial_shape, level_start_index,kwargs

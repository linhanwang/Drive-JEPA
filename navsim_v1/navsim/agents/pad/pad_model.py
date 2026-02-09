from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from .score_module.scorer import Scorer
from .traj_refiner import Traj_refiner
from .bevformer.image_encoder import ImgEncoder
from .bevformer.transformer_decoder import MLP

class PadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3

        self._backbone = ImgEncoder(config)

        self.command_num=config.command_num

        self.hist_encoding = nn.Linear(11, config.tf_d_model)

        self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)

        ref_num=config.ref_num

        shared_refiner=Traj_refiner(config)

        self._trajectory_head=nn.ModuleList([shared_refiner for _ in range(ref_num) ] )

        self.scorer = Scorer(config)

        self.b2d=config.b2d
        self.transform = self.make_transform()

    def make_transform(self):
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([normalize])

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ego_status: torch.Tensor = features["ego_status"][:,-1]
        camera_feature: torch.Tensor = features["camera_feature"]
       
        batch_size = ego_status.shape[0]

        if self.b2d:
            ego_status[:,1:3]=0

        image_feature = self._backbone(camera_feature,img_metas=features)  # b,64,64,64

        output={}

        ego_feature=self.hist_encoding(ego_status)[:,None]

        bev_feature =ego_feature+self.init_feature.weight[None]

        proposal_list = []

        for i, refine in enumerate(self._trajectory_head):
            bev_feature, proposal_list = refine(bev_feature, proposal_list,image_feature)

        proposals=proposal_list[-1]

        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        pred_logit,pred_logit2, pred_agents_states, pred_area_logit ,bev_semantic_map,agent_states,agent_labels= self.scorer(proposals, bev_feature)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels

        if pred_logit2 is not None:
            pdm_score=(torch.sigmoid(pred_logit)+torch.sigmoid(pred_logit2))[:,:,-1]/2
        else:
            pdm_score=torch.sigmoid(pred_logit)[:,:,-1]

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output




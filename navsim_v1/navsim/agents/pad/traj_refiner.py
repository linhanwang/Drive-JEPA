import torch.nn as nn
from .bevformer.bev_refiner import Bev_refiner
from .bevformer.transformer_decoder import MLP
import numpy as np

class Traj_refiner(nn.Module):
    def __init__(self,config,init_p=False):
        super().__init__()

        self.poses_num=config.num_poses
        self.state_size=3

        self.traj_bev = config.traj_bev
        self.b2d = config.b2d

        # self.init_p = init_p
        #
        # if self.init_p:
        #     self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
        if self.traj_bev:
            self.Bev_refiner=Bev_refiner(config,config.proposal_num,self.poses_num,config.traj_proposal_query)

        self.traj_decoder = MLP(config.tf_d_model, config.tf_d_ffn,  self.state_size)

    def forward(self, bev_feature,proposal_list,image_feature):

        proposals = self.traj_decoder(bev_feature).reshape(bev_feature.shape[0], -1, self.poses_num, self.state_size)

        proposal_list.append(proposals)

        if self.traj_bev:
            bev_feature = self.Bev_refiner(proposals,bev_feature,image_feature)

        return bev_feature,proposal_list



import torch
import torch.nn as nn
from ..bevformer.bev_refiner import Bev_refiner
from ..bevformer.transformer_decoder import MyTransformeDecoder,MLP
from .map_head import MapHead
import numpy as np

class Scorer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.b2d=config.b2d

        self.proposal_num=config.proposal_num
        self.score_num = 6

        self.pred_score = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)
        self.double_score=config.double_score

        if self.double_score:
            self.pred_score2 = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)

        self.agent_pred= config.agent_pred

        if self.agent_pred:
            if self.b2d:
                self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn, 2*6* 9)
            else:
                self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn,2* 40 * 9)

        self.area_pred=config.area_pred

        if self.area_pred:
            if self.b2d:
                self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 2)
            else:
                self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 5*2)

        self.bev_map=config.bev_map
        self.bev_agent=config.bev_agent

        if config.bev_agent:
            self._agent_head=MyTransformeDecoder(config,config.num_bounding_boxes,6,trajenc=False)

        if config.bev_map:
            self.map_head=MapHead(config)

    def forward(self, proposals,bev_feature):
        batch_size=len(proposals)
        p_size=proposals.shape[1]
        t_size=proposals.shape[2]

        proposal_feature = bev_feature.reshape(batch_size, p_size, t_size, -1).amax(-2)
        pred_logit = self.pred_score(proposal_feature).reshape(batch_size, -1, self.score_num)

        pred_logit2=pred_agents_states=pred_area_logit=bev_semantic_map=agent_states=agent_labels=None

        if self.double_score:
            pred_logit2 = self.pred_score2(proposal_feature).reshape(batch_size, -1, self.score_num)

        if  self.training:
            if self.area_pred:
                pred_area_logit = self.pred_area(bev_feature)

            if self.agent_pred:
                pred_agents_states = self.pred_col_agent(proposal_feature).reshape(batch_size,p_size,t_size,-1,2,9)

            if self.bev_map:
                bev_semantic_map = self.map_head(bev_feature)

            if self.bev_agent:
                agents =self._agent_head(None,bev_feature)
                agent_states = agents[:, :, :-1]
                agent_labels = agents[:, :, -1]

        return pred_logit,pred_logit2, pred_agents_states, pred_area_logit,bev_semantic_map,agent_states,agent_labels

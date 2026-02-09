from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from navsim.agents.pad.pad_model import PadModel
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.agents.pad.pad_features import PadTargetBuilder
from navsim.agents.pad.pad_features import PadFeatureBuilder
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from .score_module.compute_b2d_score import compute_corners_torch, b2d_before_score
from navsim.agents.transfuser.transfuser_loss import _agent_loss


class PadAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr: float,
            checkpoint_path: str = None,
    ):
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path

        cache_data=False

        if not cache_data:
            self._pad_model = PadModel(config)

        if not cache_data and self._checkpoint_path == "":#only for training
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.b2d = config.b2d

            self.ray=True

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                if self.b2d:
                    # self.worker = RayDistributedNoTorch(threads_per_node=8)
                    self.worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=8)
                else:
                    self.worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=16)
                self.worker_map=worker_map

            if config.b2d:
                self.train_metric_cache_paths = load_feature_target_from_pickle(
                    os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/train_fut_boxes.gz")
                self.test_metric_cache_paths = load_feature_target_from_pickle(
                    os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/val_fut_boxes.gz")
                from .score_module.compute_b2d_score import get_scores
                self.get_scores = get_scores

                map_file =os.getenv("NAVSIM_EXP_ROOT") +"/map.pkl"

                with open(map_file, 'rb') as f:
                    self.map_infos = pickle.load(f)

                self.anchors = np.load(f"{os.getenv('NAVSIM_DEVKIT_ROOT')}/data/b2d_8192.npy")
            else:
                from .score_module.compute_navsim_score import get_scores

                # metric_cache = MetricCacheLoader(Path(os.getenv("TMPFS") + "/train_ipad_metric_cache"))
                metric_cache = MetricCacheLoader(Path('/scratch/linhan/train_ipad_metric_cache'))
                # metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_ipad_metric_cache"))
                self.train_metric_cache_paths = metric_cache.metric_cache_paths
                self.test_metric_cache_paths = metric_cache.metric_cache_paths

                self.get_scores = get_scores
                
                poses = np.load("./data/8192.npy")
                self.anchors = poses[:, 4::5]

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"]
            self.load_state_dict({k.replace("agent._pad_model", "_pad_model"): v for k, v in state_dict.items()})

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[3],
            cam_l0=[3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[3],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[3],
            lidar_pc=[],
        )
    
    def get_target_builders(self) :
        return [PadTargetBuilder(config=self._config)]

    def get_feature_builders(self) :
        return [PadFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._pad_model(features)


    def compute_score(self, targets, proposals, test=True):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        target_trajectory = targets["trajectory"]
        proposals=proposals.detach()

        if self.b2d:
            data_points = b2d_before_score(self.map_infos, proposals, targets, metric_cache_paths)
        else:
            data_points = [
                {
                    "token": metric_cache_paths[token],
                    "poses": poses,
                    "test": test
                }
                for token, poses in zip(targets["token"], proposals.cpu().numpy())
            ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)
        scores_index = [res[-1] for res in all_res]

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas, scores_index

    def score_loss(self, pred_logit, pred_logit2,agents_state, pred_area_logits, target_scores, gt_states, gt_valid,
                   gt_ego_areas):

        if agents_state is not None:
            pred_states = agents_state[..., :-1].reshape(gt_states.shape)
            pred_logits = agents_state[..., -1:].reshape(gt_valid.shape)

            pred_l1_loss = F.l1_loss(pred_states, gt_states, reduction="none")[gt_valid]

            if len(pred_l1_loss):
                pred_l1_loss = pred_l1_loss.mean()
            else:
                pred_l1_loss = pred_states.mean() * 0

            pred_ce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_valid.to(torch.float32), reduction="mean")

        else:
            pred_ce_loss = 0
            pred_l1_loss = 0

        if pred_area_logits is not None:
            pred_area_logits = pred_area_logits.reshape(gt_ego_areas.shape)

            pred_area_loss = F.binary_cross_entropy_with_logits(pred_area_logits, gt_ego_areas.to(torch.float32),
                                                              reduction="mean")
        else:
            pred_area_loss = 0

        sub_score_loss = self.bce_logit_loss(pred_logit, target_scores[..., -pred_logit.shape[-1]:])  # .mean()[..., -6:]

        final_score_loss = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])  # .mean()

        if pred_logit2 is not None:
            sub_score_loss2 = self.bce_logit_loss(pred_logit2, target_scores)  # .mean()[..., -6:-1][..., -6:-1]

            final_score_loss2 = self.bce_logit_loss(pred_logit2[..., -1], target_scores[..., -1])  # .mean()

            sub_score_loss=(sub_score_loss+sub_score_loss2)/2

            final_score_loss=(final_score_loss+final_score_loss2)/2

        return sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss

    def diversity_loss(self, proposals):
        dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)

        dist = dist + (dist == 0)

        #dist[dist==0]=10000

        inter_loss = -dist.amin(1).amin(1).mean()

        return inter_loss
    
    def trajectory_loss_anchors(self, proposal_list, target_trajectory, config, scores_index): 
        trajectory_loss = 0

        min_loss_list = []
        inter_loss_list = []
        for proposals_i, idx_arr in zip(proposal_list, scores_index):
            min_loss = (
                torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1)
                .mean(-1)
                .amin(1)
                .mean()
            )

            pseudo_min_loss = 0 
            # Use scores_index to build pseudo targets from anchors
            for i, idx_arr in enumerate(scores_index):
                if idx_arr.shape[0] == 0:
                    continue

                if idx_arr.shape[0] > 2:
                    sampled_idx = np.random.choice(idx_arr, size=2, replace=False)
                else:
                    sampled_idx = idx_arr

                # self.anchors: numpy; convert to torch on same device/dtype as proposals_i
                anchors_np = self.anchors[sampled_idx]
                pseudo_targets = torch.from_numpy(anchors_np).to(
                    device=proposals_i.device, dtype=proposals_i.dtype
                )

                # proposals_i: (B, P, T, D)
                # pseudo_targets: (K, T, D)
                diff = proposals_i[i, None] - pseudo_targets[:, None]  # (B, K, P, T, D)
                pseudo_min_loss += (
                    torch.linalg.norm(diff, dim=-1, ord=1)   # (B, K, P, T)
                    .mean(-1)                                # (B, K, P)
                    .amin(-1)                                # (B, K)
                    .mean()                                  # scalar
                )
            
            min_loss += 0.5 * pseudo_min_loss / len(scores_index) 

            inter_loss = self.diversity_loss(proposals_i)
            trajectory_loss = config.prev_weight * trajectory_loss + min_loss + inter_loss * config.inter_weight

            min_loss_list.append(min_loss)
            inter_loss_list.append(inter_loss)

        return trajectory_loss, min_loss, inter_loss, min_loss_list, inter_loss_list

    def pad_loss(self,targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config  ):

        proposals = pred["proposals"]
        proposal_list = pred["proposal_list"]
        target_trajectory = targets["trajectory"]

        final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas, scores_index = self.compute_score(
            targets, proposals, test=False)
        
        trajectory_loss, min_loss, inter_loss, min_loss_list, inter_loss_list = self.trajectory_loss_anchors(proposal_list, target_trajectory, config, scores_index)

        min_loss0 = min_loss_list[0]
        inter_loss0 = inter_loss_list[0]
        # min_loss1 = min_loss_list[1]
        # inter_loss1 = inter_loss_list[1]

        if "pred_logit" in pred.keys():
            sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
                pred["pred_logit"],pred["pred_logit2"],
                pred["pred_agents_states"], pred["pred_area_logit"]
                , target_scores, gt_states, gt_valid, gt_ego_areas)
        else:
            sub_score_loss = final_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0

        if pred["agent_states"] is not None:
            agent_class_loss, agent_box_loss = _agent_loss(targets, pred, config)
        else:
            agent_class_loss = 0
            agent_box_loss = 0

        if pred["bev_semantic_map"] is not None:
            bev_semantic_loss = F.cross_entropy(pred["bev_semantic_map"], targets["bev_semantic_map"].long())
        else:
            bev_semantic_loss = 0

        loss = (
                config.trajectory_weight * trajectory_loss
                + config.sub_score_weight * sub_score_loss
                + config.final_score_weight * final_score_loss
                + config.pred_ce_weight * pred_ce_loss
                + config.pred_l1_weight * pred_l1_loss
                + config.pred_area_weight * pred_area_loss
                + config.agent_class_weight * agent_class_loss
                + config.agent_box_weight * agent_box_loss
                + config.bev_semantic_weight * bev_semantic_loss

        )

        pdm_score = pred["pdm_score"].detach()
        top_proposals = torch.argmax(pdm_score, dim=1)
        score = final_scores[np.arange(len(final_scores)), top_proposals].mean()
        best_score = best_scores.mean()

        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            'sub_score_loss': sub_score_loss,
            'final_score_loss': final_score_loss,
            'pred_ce_loss': pred_ce_loss,
            'pred_l1_loss': pred_l1_loss,
            'pred_area_loss': pred_area_loss,
            "inter_loss0": inter_loss0,
            # "inter_loss1": inter_loss1,
            "inter_loss": inter_loss,
            "min_loss0": min_loss0,
            # "min_loss1": min_loss1,
            "min_loss": min_loss,
            "score": score,
            "best_score": best_score
        }

        return loss_dict

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        return self.pad_loss(targets, pred, self._config)

    def get_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)#,weight_decay= 1e-2
        # self.lr_warmup_steps=500
        # self.lr_total_steps=20000
        # self.lr_min_ratio=1e-3

        # def lr_lambda(current_step):
        #     if current_step < self.lr_warmup_steps:
        #         return (1.0 / 3) + (current_step / self.lr_warmup_steps) * (1 - 1.0 / 3)
        #     return 1.0
        #     # return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
        #     #     1.0
        #     #     + math.cos(
        #     #         math.pi
        #     #         * min(
        #     #             1.0,
        #     #             (current_step - self.lr_warmup_steps)
        #     #             / (self.lr_total_steps - self.lr_warmup_steps),
        #     #         )
        #     #     )
        #     # )


        # scheduler = {
        #     'scheduler': LambdaLR(optimizer, lr_lambda),
        #     'interval': 'step',  # Update every step
        #     'frequency': 1
        # }

        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        # Smaller lr for vjepa encoder
        return torch.optim.Adam(self._pad_model.parameters(), lr=self._lr)#,weight_decay=1e-4

    def get_training_callbacks(self):

        checkpoint_cb = ModelCheckpoint(save_top_k=20,
                                        monitor='val/score_epoch',
                                        filename='{epoch}-{step}',
                                        mode="max"
                                        )

        return []

from time import sleep

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Tuple, Any

from navsim.agents.abstract_agent import AbstractAgent
import glob
import os
import subprocess
import shutil
import json

class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.checkpoint_file=None

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        prediction = self.agent.forward(features)
        loss = self.agent.compute_loss(features, targets, prediction)

        if type(loss) is dict:
            for key,value in loss.items():
                self.log(f"{logging_prefix}/"+key, value, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return loss["loss"]
        else:
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if 'Pad' in self.agent.name() or 'perception_based' in self.agent.name():
            features, targets = batch
            # score,best_score=self.agent.inference(features, targets)
            predictions = self.agent.forward(features)
            # if self.agent.b2d:
            #     all_res=torch.cat([predictions["trajectory"][:,None],predictions["proposals"]],dim=1)
            # else:
            all_res=predictions["trajectory"][:,None]
            final_score,best_score,proposal_scores,l2,trajectoy_scores=self.agent.compute_score(targets,all_res)
            mean_score=proposal_scores.mean()
            pdm_score=predictions["pdm_score"]
            # if self.agent.b2d:
            #     score_error=torch.abs(pdm_score - proposal_scores[:,1:]).mean()
            # else:
            pdm_score=pdm_score[torch.arange(len(pdm_score)), torch.argmax(pdm_score, dim=1)]
            score_error=torch.abs(pdm_score - proposal_scores).mean()
            logging_prefix="val"
            self.log(f"{logging_prefix}/score", final_score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/best_score", best_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/mean_score", mean_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/score_error", score_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/l2", l2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            collision=trajectoy_scores[:,0].mean()
            self.log(f"{logging_prefix}/collision", collision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            drivable_area_compliance=trajectoy_scores[:,1].mean()
            self.log(f"{logging_prefix}/dac", drivable_area_compliance, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            ego_progress=trajectoy_scores[:,2].mean()
            self.log(f"{logging_prefix}/progress", ego_progress, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            time_to_collision_within_bound=trajectoy_scores[:,3].mean()
            self.log(f"{logging_prefix}/ttc", time_to_collision_within_bound, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            comfort=trajectoy_scores[:,4].mean()
            self.log(f"{logging_prefix}/comfort", comfort, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            return final_score
        else:
            return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

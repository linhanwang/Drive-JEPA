import torch

from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.simulation_v2.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import lzma
import pickle
from navsim.evaluate.pdm_score import fast_transform_trajectory
from navsim.common.dataclasses import Trajectory
from navsim.common.utils import mean_time_every_5_calls
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)
import numpy as np
from .train_pdm_scorer import PDMScorerConfig, PDMScorer

# metric_cache_loader = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/metric_cache"))
proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)
config = PDMScorerConfig( )
scorer = PDMScorer(proposal_sampling, config)

def get_scores(args):
    return [get_sub_score(a["token"],a["poses"],a["test"]) for a in args]

def load_metric_cache(metric_cache_path) -> MetricCache:
    with lzma.open(metric_cache_path, "rb") as f:
        metric_cache = pickle.load(f)
    return metric_cache

def before_score(metric_cache: MetricCache, poses):
    initial_ego_state = metric_cache.ego_state

    trajectory_states = []
    for model_trajectory in poses:
        pred_states_2 = fast_transform_trajectory(Trajectory(model_trajectory), simulator.proposal_sampling, initial_ego_state)
        trajectory_states.append(pred_states_2)

    trajectory_states = np.stack(trajectory_states, axis=0)
    extended = np.zeros((*trajectory_states.shape[:-1], 11), dtype=trajectory_states.dtype)
    extended[:, :, :3] = trajectory_states
    
    simulated_states = simulator.simulate_proposals(extended, initial_ego_state)#32,41,11
    return simulated_states


def get_sub_score( metric_cache_path,poses,test):
    metric_cache: MetricCache = load_metric_cache(metric_cache_path)
    scores_index_path = str(metric_cache_path).replace('train_ipad_metric_cache', 'anchors_scores_index')
    scores_index = np.load(scores_index_path + '.npy')
    simulated_states = before_score(metric_cache, poses)

    initial_ego_state = metric_cache.ego_state

    final_scores=scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.pdm_progress
    )

    num_col=2

    key_agent_corners = np.zeros([len(final_scores), scorer.proposal_sampling.num_poses ,num_col, 4, 2])
    key_agent_labels = np.zeros([len(final_scores), scorer.proposal_sampling.num_poses ,num_col],dtype=bool)
    ego_areas = scorer._ego_areas[:,1:,1:]

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, :]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, :]
    #driving_direction_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, :  ]

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, :]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, :]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, :]


    scores=np.stack([no_at_fault_collisions,drivable_area_compliance,#driving_direction_compliance,
                     ego_progress,time_to_collision_within_bound,comfort,final_scores
                     ],axis=-1)#[:,None]

    if not test:
        for i in range(len(scores)):
            # proposal_collided_track_ids=scorer.proposal_collided_track_ids[i]
            proposal_fault_collided_track_ids = scorer.proposal_fault_collided_track_ids[i]
            # temp_collided_track_ids=scorer.temp_collided_track_ids[i]

            if len(proposal_fault_collided_track_ids):
                col_token=proposal_fault_collided_track_ids[0]
                collision_time_idcs = int(scorer._collision_time_idcs[i])+1

                for time_idx in range(1,collision_time_idcs):
                    if  col_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][time_idx-1,0] = True
                        key_agent_corners[i][time_idx-1,0]=np.array(scorer._observation[time_idx][col_token].boundary.xy).T[:4]

            ttc_collided_track_ids = scorer.ttc_collided_track_ids[i]

            if len(ttc_collided_track_ids):
                ttc_token=ttc_collided_track_ids[0]
                ttc_time_idcs = int(scorer._ttc_time_idcs[i])+1

                for time_idx in range(1,ttc_time_idcs):
                    if  ttc_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][time_idx-1,1] = True
                        key_agent_corners[i][time_idx-1,1]=np.array(scorer._observation[time_idx][ttc_token].boundary.xy).T[:4]

        theta = initial_ego_state.rear_axle.heading
        origin_x = initial_ego_state.rear_axle.x
        origin_y = initial_ego_state.rear_axle.y

        c, s = np.cos(theta), np.sin(theta)
        mat = np.array([[c, -s],
                        [s, c]])

        key_agent_corners[...,0]-=origin_x
        key_agent_corners[...,1]-=origin_y

        key_agent_corners=key_agent_corners.dot(mat)

    return scores,key_agent_corners,key_agent_labels,ego_areas, scores_index

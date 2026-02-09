import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.creation import linestrings
from shapely import Point, creation
from shapely.strtree import STRtree
import shapely
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.observation.idm.utils import is_agent_behind, is_track_stopped
from shapely import LineString, Polygon
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
    
def b2d_before_score(map_infos, proposals, targets, metric_cache_paths):
    data_points = []
    
    target_trajectory = targets["trajectory"]
    lidar2worlds=targets["lidar2world"]
    
    all_proposals = torch.cat([proposals, target_trajectory[:,None]], dim=1)
    all_proposals_xy=all_proposals[:, :,:, :2]
    all_proposals_heading=all_proposals[:, :,:, 2:]
    all_pos = all_proposals_xy.reshape(len(target_trajectory),-1, 2)
    mid_points = (all_pos.amax(1) + all_pos.amin(1)) / 2
    dists = torch.linalg.norm(all_pos - mid_points[:,None], dim=-1).amax(1) + 5

    xyz = torch.cat(
        [mid_points[..., :2], torch.zeros_like(mid_points[..., :1]), torch.ones_like(mid_points[..., :1])], dim=-1)
    xys = torch.einsum("nij,nj->ni", lidar2worlds, xyz)[:, :2]
    vel=torch.cat([all_proposals_xy[:, :,:1], all_proposals_xy[:,:, 1:] - all_proposals_xy[:,:, :-1]],dim=2)/ 0.5
    proposals_05 = torch.cat([all_proposals_xy + vel*0.5, all_proposals_heading], dim=-1)
    proposals_1 = torch.cat([all_proposals_xy + vel*1, all_proposals_heading], dim=-1)
    proposals_ttc = torch.stack([all_proposals, proposals_05,proposals_1], dim=3)

    ego_corners_ttc = compute_corners_torch(proposals_ttc.reshape(-1, 3)).reshape(proposals_ttc.shape[0],proposals_ttc.shape[1], proposals_ttc.shape[2], proposals_ttc.shape[3],  4, 2)
    ego_corners_center = torch.cat([ego_corners_ttc[:,:,:,0], all_proposals_xy[:, :, :, None]], dim=-2)
    ego_corners_center_xyz = torch.cat(
        [ego_corners_center, torch.zeros_like(ego_corners_center[..., :1]), torch.ones_like(ego_corners_center[..., :1])], dim=-1)
    global_ego_corners_centers = torch.einsum("nij,nptkj->nptki", lidar2worlds, ego_corners_center_xyz)[..., :2]

    accs = torch.linalg.norm(vel[:,:, 1:] - vel[:,:, :-1], dim=-1) / 0.5
    turning_rate=torch.abs(torch.cat([all_proposals_heading[:, :,:1,0]-np.pi/2, all_proposals_heading[:,:, 1:,0]-all_proposals_heading[:,:, :-1,0]],dim=2)) / 0.5
    comforts = (accs[:,:-1] < accs[:,-1:].max()).all(-1) & (turning_rate[:,:-1] < turning_rate[:,-1:].max()).all(-1)

    for key, value in map_infos.items():
        map_infos[key] = torch.tensor(value).to(target_trajectory.device)

    for token, town_name, proposal,target_traj, comfort, dist, xy,global_conners,local_corners in zip(targets["token"], targets["town_name"], proposals.cpu().numpy(),  target_trajectory.cpu().numpy(), comforts.cpu().numpy(), dists.cpu().numpy(), xys, global_ego_corners_centers,ego_corners_ttc.cpu().numpy()):
        all_lane_points = map_infos[town_name[:6]]
        dist_to_cur = torch.linalg.norm(all_lane_points[:,:2] - xy, dim=-1)
        nearby_point = all_lane_points[dist_to_cur < dist]

        if nearby_point.numel() == 0:
            # No lane points found within dist: avoid argmin on empty tensors
            T = global_conners.shape[0]
            P = global_conners.shape[1]
            on_road_all = torch.zeros((T, P), dtype=torch.bool, device=global_conners.device)
            on_route_all = torch.zeros((T, P), dtype=torch.bool, device=global_conners.device)
            batch_multiple_lanes_mask = torch.zeros((T, P), dtype=torch.bool, device=global_conners.device)

            ego_areas = torch.stack([batch_multiple_lanes_mask, on_road_all, on_route_all], dim=-1)

            data_dict = {
                "fut_box_corners": metric_cache_paths[token],
                "_ego_coords": local_corners,
                "target_traj": target_traj,
                "proposal":proposal,
                "comfort": comfort,
                "ego_areas": ego_areas.cpu().numpy(),
                "token": token,
            }
            data_points.append(data_dict)
            continue

        lane_xy = nearby_point[:, :2]
        lane_width = nearby_point[:, 2]
        lane_id = nearby_point[:, -1]

        dist_to_lane = torch.linalg.norm(global_conners[None] - lane_xy[:, None, None, None], dim=-1)
        on_road = dist_to_lane < lane_width[:, None, None, None]
        on_road_all = on_road.any(0).all(-1)
        nearest_lane = torch.argmin(dist_to_lane - lane_width[:, None, None,None], dim=0)
        nearest_lane_id=lane_id[nearest_lane]
        center_nearest_lane_id=nearest_lane_id[:,:,-1]
        nearest_road_id = torch.round(center_nearest_lane_id)
        target_road_id = torch.unique(nearest_road_id[-1])

        on_route_all = torch.isin(nearest_road_id, target_road_id)
        corner_nearest_lane_id=nearest_lane_id[:,:,:-1]
        batch_multiple_lanes_mask = (corner_nearest_lane_id!=corner_nearest_lane_id[:,:,:1]).any(-1)

        on_road_all=on_road_all==on_road_all[-1:]

        ego_areas=torch.stack([batch_multiple_lanes_mask,on_road_all,on_route_all],dim=-1)

        data_dict = {
            "fut_box_corners": metric_cache_paths[token],
            "_ego_coords": local_corners,
            "target_traj": target_traj,
            "proposal":proposal,
            "comfort": comfort,
            "ego_areas": ego_areas.cpu().numpy(),
            "token": token,
        }
        data_points.append(data_dict)
    return data_points

def compute_corners_torch(proposals):

    headings= proposals[...,2]
    cos_yaw = torch.cos(headings)
    sin_yaw = torch.sin(headings)

    x = proposals[...,0]+0.39*cos_yaw
    y = proposals[...,1]+0.39*sin_yaw
    half_length=  2.042+torch.zeros_like(headings)
    half_width = 0.925+torch.zeros_like(headings)

    cos_yaw=cos_yaw[...,None]
    sin_yaw=sin_yaw[...,None]

    # Compute the four corners
    corners_x = torch.stack([half_length, -half_length, -half_length, half_length],dim=-1)
    corners_y = torch.stack([half_width, half_width, -half_width, -half_width],dim=-1)

    # Rotate corners by yaw
    rot_corners_x = cos_yaw * corners_x + (-sin_yaw) * corners_y
    rot_corners_y = sin_yaw * corners_x + cos_yaw * corners_y

    # Translate corners to the center of the bounding box
    corners = torch.stack((rot_corners_x + x[...,None], rot_corners_y + y[...,None]), dim=-1)

    return corners
    # FRONT_LEFT = 0
    # REAR_LEFT = 1
    # REAR_RIGHT = 2
    # FRONT_RIGHT = 3


def get_collision_type(
        state,
        ego_polygon: Polygon,
        tracked_object_polygon: Polygon,
        track_speed,
        track_heading,
        stopped_speed_threshold: float = 5e-02,
):
    ego_speed = state[-1]

    is_ego_stopped = float(ego_speed) <= stopped_speed_threshold

    center_point = tracked_object_polygon.centroid

    tracked_object_center = StateSE2(center_point.x, center_point.y, track_heading)

    x=state[0]
    y=state[1]
    ego_heading=state[2]

    ego_rear_axle_pose: StateSE2 = StateSE2(x,y,ego_heading)

    # Collisions at (close-to) zero ego speed
    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION

    # Collisions at (close-to) zero track speed
    elif track_speed <= stopped_speed_threshold:
        collision_type = CollisionType.STOPPED_TRACK_COLLISION

    # Rear collision when both ego and track are not stopped
    elif is_agent_behind(ego_rear_axle_pose, tracked_object_center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION

    # Front bumper collision when both ego and track are not stopped
    elif LineString(
            [
                ego_polygon.exterior.coords[0],
                ego_polygon.exterior.coords[3],
            ]
    ).intersects(tracked_object_polygon):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION

    # Lateral collision when both ego and track are not stopped
    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION

    return collision_type


def evaluate_coll( fut_box_corners,_ego_coords,_ego_areas):
    n_future = _ego_coords.shape[1]
    _num_proposals=_ego_coords.shape[0]
    fut_mask=fut_box_corners.any(-1).any(-1)

    node_capacity=10

    _ego_polygons = shapely.creation.polygons(_ego_coords)

    proposal_fault_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }
    proposal_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }
    temp_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }
    ttc_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }

    key_agent_corners = np.zeros([_num_proposals,6,2,  4, 2])
    key_agent_labels = np.zeros([_num_proposals, 6,2],dtype=bool)
    collision_all = np.zeros([_num_proposals,n_future],dtype=bool)
    ttc_collision_all = np.zeros([_num_proposals,n_future],dtype=bool)

    # ego_pos=_ego_coords[:,:,0].mean(-2)
    # ego_vel=(_ego_coords[:,:,1,0]-_ego_coords[:,:,0,0])*2
    #
    # speeds=np.linalg.norm(ego_vel,axis=-1)
    #
    # # Compute the vector along the front edge
    # front_edge = _ego_coords[:,:, 0,0, :] - _ego_coords[:,:, 0, 1, :]
    #
    # # Compute heading angle relative to x-axis
    # ego_headings = np.arctan2(front_edge[..., 1], front_edge[..., 0])
    #
    # ego_state=np.concatenate([ego_pos,ego_headings[...,None],speeds[...,None]],axis=-1)

    # fut_box_center=fut_box_corners.mean(-2)

    # track_object_vel = (fut_box_center[:,1:]-fut_box_center[:,:-1])*2
    #
    # track_object_vel=np.concatenate([track_object_vel[:,:1],track_object_vel],axis=1)
    #
    # track_object_speeds=np.linalg.norm(track_object_vel,axis=-1)

    # Compute the vector along the front edge
    # object_front_edge = fut_box_corners[..., 0, :] - fut_box_corners[..., 1, :]
    #
    # # Compute heading angle relative to x-axis
    # track_object_headings = np.arctan2(object_front_edge[..., 1], object_front_edge[..., 0])

    for time_idx in range(n_future):
        geometries=fut_box_corners[:,time_idx][fut_mask[:,time_idx]]
        _geometries = [Polygon(geometry) for geometry in geometries]
        _str_tree = STRtree(_geometries, node_capacity)
        ego_polygons = _ego_polygons[:, time_idx,0]

        intersecting = _str_tree.query(ego_polygons, predicate="intersects")

        token_list=np.arange(len(fut_box_corners))[fut_mask[:,time_idx]]

        for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
            token=token_list[geometry_idx]
            if token in proposal_collided_track_ids[proposal_idx] or len(proposal_fault_collided_track_ids[proposal_idx]):
                continue

            # ego_in_multiple_lanes_or_nondrivable_area = (
            #         _ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
            #         or _ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
            # )
            #
            # tracked_object_polygon = _geometries[geometry_idx]
            #
            # # classify collision
            # collision_type: CollisionType = get_collision_type(
            #     ego_state[proposal_idx][time_idx],
            #     ego_polygons[proposal_idx],
            #     tracked_object_polygon,
            #     track_object_speeds[token][time_idx],
            #     track_object_headings[token][time_idx],
            #
            # )
            # collisions_at_stopped_track_or_active_front: bool = collision_type in [
            #     CollisionType.ACTIVE_FRONT_COLLISION,
            #     CollisionType.STOPPED_TRACK_COLLISION,
            # ]
            # collision_at_lateral: bool = collision_type == CollisionType.ACTIVE_LATERAL_COLLISION

            # 1. at fault collision
            # if collisions_at_stopped_track_or_active_front or (
            #         ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
            # ):
            proposal_fault_collided_track_ids[proposal_idx].append(token)
            collision_all[proposal_idx][time_idx]=True
            key_agent_labels[proposal_idx][:time_idx+1,0]=fut_mask[token][:time_idx+1]
            key_agent_corners[proposal_idx][:time_idx+1,0]=fut_box_corners[token][:time_idx+1]
            # else:  # 2. no at fault collision
            #     proposal_collided_track_ids[proposal_idx].append(token)

        for ttc in [1,2]:
            ego_polygons = _ego_polygons[:, time_idx,ttc]

            intersecting = _str_tree.query(ego_polygons, predicate="intersects")

            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = token_list[geometry_idx]
                if (token in temp_collided_track_ids[proposal_idx] or len(ttc_collided_track_ids[proposal_idx]) #or (speeds[proposal_idx, time_idx] < 5e-02)
                ):
                    continue

                # ego_in_multiple_lanes_or_nondrivable_area = (
                #         _ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                #         or _ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                # )
                #
                # state=ego_state[proposal_idx][time_idx]
                # ego_heading = np.arctan2(state[-1], state[-2])
                #
                # ego_rear_axle: StateSE2 = StateSE2(state[0], state[1], ego_heading)
                # tracked_object_polygon = _geometries[geometry_idx]
                #
                # centroid = tracked_object_polygon.centroid
                # track_heading = track_object_headings[token][time_idx]
                # track_state = StateSE2(centroid.x, centroid.y, track_heading)

                # TODO: fix ego_area for intersection
                # if is_agent_ahead(ego_rear_axle, track_state) or (
                #         ego_in_multiple_lanes_or_nondrivable_area
                #         and not is_agent_behind(ego_rear_axle, track_state)
                # ):
                ttc_collided_track_ids[proposal_idx].append(token)
                ttc_collision_all[proposal_idx][time_idx]=True
                key_agent_labels[proposal_idx][:time_idx+1,1]=fut_mask[token][:time_idx+1]
                key_agent_corners[proposal_idx][:time_idx+1,1]=fut_box_corners[token][:time_idx+1]
                # else:
                #     temp_collided_track_ids[proposal_idx].append(token)

    return collision_all,ttc_collision_all,key_agent_corners[:-1],key_agent_labels[:-1]

def get_scores(args):

    return [get_sub_score(a["fut_box_corners"],a["_ego_coords"],a["proposal"],a["target_traj"],a["comfort"],a["ego_areas"], a["token"]) for a in args]

def get_sub_score(fut_box_corners,_ego_coords,proposals,target_traj,comfort,ego_areas, token):
    
    collsions,ttc_collision,key_agent_corners,key_agent_labels=evaluate_coll(fut_box_corners,_ego_coords,ego_areas)

    collsions=collsions[:-1] & (~collsions[-1:])  #collsion=True and gt_collision =False

    ttc_collision=ttc_collision[:-1] & (~ttc_collision[-1:])  #collsion=True and gt_collision =False

    collision=1-collsions.any(-1)

    ttc=1-ttc_collision.any(-1)

    on_road_all=ego_areas[:-1,:,1]
    on_route_all=ego_areas[:-1,:,2]

    drivable_area_compliance=on_road_all.all(-1) & on_route_all.any(-1)

    ego_areas=np.stack([on_road_all,on_route_all],axis=-1)

    #l2 = np.linalg.norm(proposals[:,:, :2] - target_traj[None, : ,:2], axis=-1).mean(-1)

    # l2_score=l2-100*multiplicate_metric_scores
    #
    # sort_score=np.sort(l2_score)
    #
    # progress =np.zeros([len(proposals)])
    #
    # progress[sort_score[0]==l2_score]=1
    # progress[sort_score[1]==l2_score]=1

    target_line=np.concatenate([np.zeros([1,2]),target_traj[...,:2]])

    centerline=linestrings(target_line)

    target_progress = centerline.project(Point(target_line[-1]))

    raw_progress = np.ones([len(proposals)])

    for proposal_idx,proposal in enumerate(proposals[...,:2]):
        end_point = Point(proposal[-1])
        proj_progress=centerline.project(end_point)
        if proj_progress==target_progress:
            proj_progress=proj_progress+np.linalg.norm(proposal[-1]-target_traj[-1][:2])

        raw_progress[proposal_idx] = proj_progress #clip by max target progress

    raw_progress = np.clip(raw_progress, a_min=0, a_max=None)

    multiplicate_metric_scores=collision*drivable_area_compliance

    max_raw_progress = np.maximum(raw_progress, target_progress)+0.01

    min_raw_progress = np.minimum(raw_progress, target_progress)+0.01

    progres_ratio=min_raw_progress/max_raw_progress

    progress=multiplicate_metric_scores*progres_ratio
    #raw_progress=multiplicate_metric_scores*raw_progress

    # progress_distance_threshold=5
    #
    # fast_mask = max_raw_progress > progress_distance_threshold
    #
    # progress = np.ones([len(raw_progress)], dtype=np.float64)
    #
    # progress[fast_mask] = raw_progress[fast_mask] / max_raw_progress[fast_mask]
    # progress[~fast_mask] = multiplicate_metric_scores[~fast_mask]

    # proposals_xy=np.concatenate([np.zeros_like(proposals[:,:1,:2]),proposals[:,:,:2]],axis=1)
    #
    # vel=(proposals_xy[:,1:,:2]-proposals_xy[:,:-1,:2])/0.5
    #
    # acc=np.linalg.norm(vel[:,1:]-vel[:,:-1],axis=-1)/0.5

    # angle=np.arctan2(vel[:,:,1], vel[:,:,0])

    # heading=np.concatenate([np.zeros_like(angle[:,:1])+np.pi/2,angle],axis=1)

    # yaw_rate=(heading[:,1:]-heading[:,:-1])/0.5
    
    # yaw_accel=(yaw_rate[:,1:]-yaw_rate[:,:-1])/0.5

    # desired_speed=np.linalg.norm(vel,axis=-1).mean(-1)
    
    # comfort=(acc<10).all(-1) #& (desired_speed<15) & (np.abs(yaw_rate)<2).all(-1) & (np.abs(yaw_accel)<4).all(-1)


    final_scores=collision*drivable_area_compliance*(ttc*5/12+progress*5/12+comfort*2/12)

    target_scores=np.stack([collision,drivable_area_compliance,progress,ttc,comfort,final_scores],axis=-1)

    #print(target_scores[0])
    # if target_scores.mean()!=1:
    #     print(1)
    
    score_index = np.load(f'/scratch/linhan/b2d_anchors_scores_index_8192/{token}.npy')

    return target_scores,key_agent_corners,key_agent_labels,ego_areas,score_index 

from dataclasses import dataclass

@dataclass
class PadConfig:
    b2d = True

    ref_num: int=4

    traj_bev: bool=True
    traj_proposal_query: bool=True

    double_score: bool=False
    agent_pred: bool=True
    area_pred: bool=True
    bev_map: bool=False
    bev_agent: bool=False
    
    proposal_num: int = 64
    point_cloud_range= [-32, -32, 0.0, 32, 32,4.0]
    num_points_in_pillar: int=4

    half_length = 2.042 
    half_width= 0.925 
    rear_axle_to_center =0.39
    lidar_height=1.84

    num_poses=6
    command_num=7

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.1
    num_bev_layers: int = 1
    image_architecture: str = "resnet34"

    # loss weights
    trajectory_weight: float = 1
    inter_weight: float =  0
    sub_score_weight: int = 0
    final_score_weight: int = 1
    pred_ce_weight: int = 1
    pred_l1_weight: int = 0.1
    pred_area_weight: int = 2
    prev_weight: int = 0.1
    agent_class_weight: float = 1.0
    agent_box_weight: float = 0.1
    bev_semantic_weight: float = 1.0

    # detection
    num_bounding_boxes: int = 30

    num_bev_classes = 15
    bev_features_channels: int = 64
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    latent: bool = False
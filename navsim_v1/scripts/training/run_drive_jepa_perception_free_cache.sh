#!/bin/bash

source scripts/env.sh

python navsim/planning/script/run_dataset_caching.py \
  agent=drive_jepa_perception_free_b_agent \
  agent.pretrain_pt_path="/home/linhan/yinlin/projects/navsim_workspace/vjepa2_ckpts/vitl_merge_3dataset/e50.pt" \
  agent.image_architecture="vit_large" \
  agent.front_only=true \
  agent.freeze_encoder=false \
  agent.double_image=true \
  experiment_name=cache_agent \
  train_test_split=navtrain \
  worker.max_workers=4 \
  worker=single_machine_thread_pool  \
  cache_path="/scratch/linhan/train_drive_jepa_perception_free_b_cache_v1" 

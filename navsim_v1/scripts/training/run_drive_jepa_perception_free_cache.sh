#!/bin/bash

export NAVSIM_DEVKIT_ROOT=$(pwd)

python navsim/planning/script/run_dataset_caching.py \
  agent=drive_jepa_perception_free_agent \
  agent.pretrain_pt_path="${NAVSIM_EXP_ROOT}/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt" \
  agent.image_architecture="vit_large" \
  agent.front_only=true \
  agent.freeze_encoder=false \
  agent.double_image=true \
  experiment_name=cache_agent \
  train_test_split=navtrain \
  worker.max_workers=4 \
  worker=single_machine_thread_pool  \
  cache_path="${NAVSIM_EXP_ROOT}/train_drive_jepa_perception_free_cache" 

#!/bin/bash

export NAVSIM_DEVKIT_ROOT=$(pwd)

python navsim/planning/script/run_dataset_caching.py \
  agent=drive_jepa_perception_based_agent \
  experiment_name=cache_agent \
  train_test_split=navtrain \
  worker.max_workers=4 \
  worker=single_machine_thread_pool  \
  cache_path="${NAVSIM_EXP_ROOT}/train_drive_jepa_perception_based_cache" 

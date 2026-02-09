#!/bin/bash

export NAVSIM_DEVKIT_ROOT=$(pwd)

CHECKPOINT='/home/linhan/navsim_workspace/exp/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl.ckpt'
# Or you can use the ckpt trained by yourself.

TRAIN_TEST_SPLIT=navtest

python -u $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=drive_jepa_perception_based_agent \
agent.config.latent=False \
worker=single_machine_thread_pool \
worker.max_workers=4 \
worker.use_process_pool=true \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=eval_reproduce_drive_jepa_perception_based_agent

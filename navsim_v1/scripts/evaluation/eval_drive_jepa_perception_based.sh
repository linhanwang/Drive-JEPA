#!/bin/bash
source scripts/env.sh

CHECKPOINT='/home/linhan/projects/navsim_workspace/exp/ke/train_ipad_vjepa_baseline_p32_topk_anchors_4/12.09_19.43/checkpoints/best_epoch\=14.ckpt'

TRAIN_TEST_SPLIT=navtest

python -u $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=drive_jepa_perception_based_agent \
agent.config.latent=False \
worker=single_machine_thread_pool \
worker.max_workers=4 \
worker.use_process_pool=true \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=eval_test
# experiment_name=eval_train_ipad_vjepa_p32_topk_anchors_4_e14

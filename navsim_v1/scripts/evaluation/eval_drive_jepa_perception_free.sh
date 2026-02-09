#!/bin/bash
source scripts/env.sh

CHECKPOINT="${NAVSIM_EXP_ROOT}/train_vjepa_arc_merge3_b32/2025.10.15.15.06.24/checkpoints/best_epoch\=32.ckpt"

TRAIN_TEST_SPLIT=navtest

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=drive_jepa_perception_free_agent \
agent.pretrain_pt_path="/home/linhan/projects/navsim_workspace/vjepa2_ckpts/vitl_merge_3dataset/e50.pt" \
agent.image_architecture="vit_large" \
agent.num_keyval=129 \
agent.front_only=true \
agent.tf_dropout=0.0 \
agent.freeze_encoder=false \
agent.double_image=true \
worker=single_machine_thread_pool \
worker.max_workers=4 \
worker.use_process_pool=true \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=eval_drive_jepa_perception_free_e32

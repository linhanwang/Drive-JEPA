#!/bin/bash

export NAVSIM_DEVKIT_ROOT=$(pwd)

CHECKPOINT="${NAVSIM_EXP_ROOT}/Drive-JEPA-cache/drive_jepa_perception_free_agent_vitl.ckpt"
# Or you can use the ckpt trained by yourself.

TRAIN_TEST_SPLIT=navtest

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=drive_jepa_perception_free_agent \
agent.pretrain_pt_path="${NAVSIM_EXP_ROOT}/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt" \
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
experiment_name=eval_reproduce_drive_jepa_perception_free_agent

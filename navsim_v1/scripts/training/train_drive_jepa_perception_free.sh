#!/bin/bash

export NAVSIM_DEVKIT_ROOT=$(pwd)

TRAIN_TEST_SPLIT=navtrain

torchrun --standalone --nproc_per_node=gpu $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=drive_jepa_perception_free_agent \
        agent.pretrain_pt_path="${NAVSIM_EXP_ROOT}/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt" \
        agent.image_architecture="vit_large" \
        agent.lr=1e-4 \
        agent.tf_dropout=0.0 \
        agent.num_keyval=129 \
        agent.front_only=true \
        agent.freeze_encoder=false \
        agent.double_image=true \
        dataloader.params.batch_size=32 \
        experiment_name=train_drive_jepa_perception_free_agent_l1_in_3  \
        train_test_split=$TRAIN_TEST_SPLIT \
        trainer.checkpoint.monitor=val/loss_epoch \
        trainer.checkpoint.mode=min \
        trainer.params.max_epochs=40 \
        trainer.params.precision=bf16  \
        cache_path="${NAVSIM_EXP_ROOT}/train_drive_jepa_perception_free_cache" \
        use_cache_without_dataset=True  \
        force_cache_computation=False 

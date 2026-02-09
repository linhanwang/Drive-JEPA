#!/bin/bash
source scripts/env.sh

TRAIN_TEST_SPLIT=navtrain

torchrun --standalone --nproc_per_node=gpu $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=drive_jepa_perception_based_agent \
        dataloader.params.batch_size=32 \
        experiment_name=train_drive_jepa_perception_based_agent  \
        train_test_split=$TRAIN_TEST_SPLIT \
        trainer.params.max_epochs=20 \
        trainer.params.strategy=ddp_find_unused_parameters_true \
        cache_path="${NAVSIM_EXP_ROOT}/train_drive_jepa_perception_based_cache_v1/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False 

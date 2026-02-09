#!/bin/bash
source scripts/env.sh

TRAIN_TEST_SPLIT=navtrain

torchrun --standalone --nproc_per_node=gpu $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=drive_jepa_perception_free_b_agent \
        agent.pretrain_pt_path="/home/linhan/yinlin/projects/navsim_workspace/vjepa2_ckpts/vitl_merge_3dataset/e50.pt" \
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
        cache_path="/scratch/linhan/train_drive_jepa_perception_free_b_cache_v1/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False 

<div align="center">

  <h1 align="center">Drive-JEPA: Video JEPA Meets Multimodal Trajectory
Distillation for End-to-End Driving</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2601.22032-b31b1b)](https://arxiv.org/abs/2601.22032)
[![Drive-JEPA Data](https://img.shields.io/badge/huggingface-Drive--JEPA-orange?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/LinhanWang/Drive-JEPA)


</div>

<div align="center">
<img width="600" alt="image" src="assets/teaser.png">
<p>Drive-JEPA outperforms prior methods in both perception-free and perception-based settings.</p>
</div>

## Table of Contents
- [📖 Abstract](#abstract)
- [🚀 Pipeline](#pipeline)
- [🖼️ Visualization](#visualization)
- [🗓️ TODO](#todo)
- [Data and weights](#data-and-weights)
- [Environment preparation](#environment-preparation)
- [Cache features](#cache-features)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## 📖 Abstract
End-to-end autonomous driving increasingly leverages self-supervised video pretraining to learn transferable planning representations. However, pretraining video world models for scene understanding has so far brought only limited improvements. This limitation is compounded by the inherent ambiguity of driving: each scene typically provides only a single human trajectory, making it difficult to learn multimodal behaviors. In this work, we propose Drive-JEPA, a framework that integrates Video Joint-Embedding Predictive Architecture (V-JEPA) with multimodal trajectory distillation for end-to-end driving. First, we adapt V-JEPA for end-to-end driving, pretraining a ViT encoder on large-scale driving videos to produce predictive representations aligned with trajectory planning. Second, we introduce a proposal-centric planner that distills diverse simulator-generated trajectories alongside human trajectories, with a momentum-aware selection mechanism to promote stable and safe behavior. When evaluated on NAVSIM, the V-JEPA representation combined with a simple transformer-based decoder outperforms prior methods by 3 PDMS in the perception-free setting. The complete Drive-JEPA framework achieves 93.3 PDMS on v1 and 87.8 EPDMS on v2, setting a new state-of-the-art.

## 🚀 Pipeline

<div align="center">
<img width="800" alt="image" src="assets/figure_pipeline.png">
<p>Overview of the Drive-JEPA architecture. Driving Video Pretraining learns a ViT encoder from large-scale driving videos
using the self-supervised V-JEPA objective. Given the pretrained features, Waypoint-anchored Proposal Generation efficiently produces
multiple trajectory proposals, whose distribution is guided by Multimodal Trajectory Distillation. Finally, Momentum-aware Trajectory
Selection picks the final trajectory by accounting for cross-frame comfort.</p>
</div>

## 🖼️ Visualization
<div align="center">
<img width="800" alt="image" src="assets/more_mtd_demo.png">
<p>Bird's eye view of proposals. Without Multimodal Trajectory Distillation (MTD), the proposals collapse into one mode. With
MTD, the proposals show multimodal distribution.</p>
</div>

## 🗓️ TODO
- [✔] Release code and checkpoints for navsim v1
- [ ] Release code and checkpoints for navsim v2
- [ ] Release code and checkpoints for Bench2Drive

## Data and weights

Please download the navsim dataset and organize the data in the same way as [HERE](https://github.com/autonomousvision/navsim/blob/main/docs/install.md).

```bash
bash download/download_maps.sh
bash download/download_test.sh
bash download/download_navtrain.sh
```

Set the required environment variables, by adding the following to your ~/.bashrc file Based on the structure above, the environment variables need to be defined as:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

Then download pretrained weights and cached data.

```bash
huggingface-cli download LinhanWang/Drive-JEPA --repo-type dataset --local-dir $NAVSIM_EXP_ROOT/Drive-JEPA-cache
```

After decompressing the tar files, please change the file paths in csv files under metric_cache/metadata and train_metric_cache/metadata.

## Environment preparation

```bash
conda create -n drive-jepa python=3.9
conda activate drive-jepa 
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cd navsim_v1 && pip install -e . # Or cd navsim_v2 && pip install -e . 
```

The tutorial below using perception-based setting on navsim v1 as an example. The process is similar for perception-free setting and navsim v2.

## Cache features

```bash
bash scripts/training/run_drive_jepa_perception_based_cache.sh
# Or bash scripts/training/run_drive_jepa_perception_free_cache.sh
```

## Training

We tested training on two L40S/A30/V100 GPUs.
```bash
bash scripts/training/train_drive_jepa_perception_based.sh
# Or bash scripts/training/train_drive_jepa_perception_free.sh
```

## Evaluation

You can use either the pretrained weights under $NAVSIM_EXP_ROOT/Drive-JEPA-cache or your own checkpoints under $NAVSIM_EXP_ROOT.

```bash
bash scripts/evaluation/eval_drive_jepa_perception_based.sh
# Or bash scripts/evaluation/eval_drive_jepa_perception_free.sh
```

## Citation

If you find Drive-JEPA useful in your research or application, please cite using this BibTex:

```
@article{wang2026drive,
  title={Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving},
  author={Wang, Linhan and Yang, Zichong and Bai, Chen and Zhang, Guoxiang and Liu, Xiaotong and Zheng, Xiaoyin and Long, Xiao-Xiao and Lu, Chang-Tien and Lu, Cheng},
  journal={arXiv preprint arXiv:2601.22032},
  year={2026}
}
```

## Acknowledgements

We borrowed code from [NAVSIM](https://github.com/autonomousvision/navsim), [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive), [VJEPA 2](https://github.com/facebookresearch/vjepa2), [iPad](https://github.com/Kguo-cs/iPad) and [LAW](https://github.com/BraveGroup/LAW). Thanks for their contribution to the community.

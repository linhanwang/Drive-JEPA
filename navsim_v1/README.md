# Install
```bash
conda create -n drive-jepa python=3.9
conda activate drive-jepa 
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

# Set environment variable
set the environment variable based on where you place the PAD directory. 
```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/pad_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/pad_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/pad_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/pad_workspace/dataset"
```


# Navsim training and evaluation
1. download the navtrain dataset and map as [Navsim](https://github.com/autonomousvision/navsim) 
```bash
bash download/download_maps.sh
bash download/download_navtrain.sh
bash download/download_navtest.sh
```
Put the downloaded maps in "dataset/maps", and dataset in "dataset/navsim_logs" and "dataset/sensor_blobs" 

2. cache training data and metric
```bash
python navsim/planning/script/run_training_metric_caching.py
python navsim/planning/script/run_dataset_caching.py
```
Change "cache_data=True" in [pad_agent](navsim/agents/pad/pad_agent.py) during running "run_dataset_caching.py".

3. train navsim model
```bash
python navsim/planing/script/run_training.py
```
4. test navsim model


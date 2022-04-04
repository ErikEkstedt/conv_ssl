# Continuous Conversational SSL

Model training for the paper [Voice Activity Projection: Self-supervised Learning of Turn-taking Events]() using [pytorch_lightning](https://pytorch-lightning.readthedocs.io/en/latest/).


## Installation
* Create conda env: `conda create -n conv_ssl python=3.9`
  - source env: `conda source conv_ssl`
* PyTorch: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
* Dependencies: `pip install -r requirements.txt`
* install conv_ssl: `pip install -e .`

* DATASET
  - WARNING: Requires [Switchboard](https://catalog.ldc.upenn.edu/LDC97S62) audio files
  * Install [datasets_turntaking](https://github.com/ErikEkstedt/datasets_turntaking)
    - clone repo `git clone https://github.com/ErikEkstedt/datasets_turntaking.git`
    - cd to repo, and install dependencies: `pip install -r requirements.txt`
    - install repo: `pip install -e .`

## Pretrained Encoders
The pretrained encoder checkpoint is downloaded from the original repo (or from torch.hub through torchaudio).

* [CPC](https://github.com/facebookresearch/CPC_audio)
  - requires installation of source
  - that is clone [CPC_audio](https://github.com/facebookresearch/CPC_audio)
    - cd to repo
    - Install dependencies (see repository) but probably you'll need
      - `pip install cython`
    - run: `python setup.py develop`  (again see original implementation)
  - **automatically** downloads checkpoints

## Docker
* Requires [Nvidia-Docker]() for gpu support.
  * [Nvidia Docker Github](https://github.com/NVIDIA/nvidia-docker)
  * [github.io docs](https://nvidia.github.io/nvidia-docker/)
  * [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
* sudo may be required for default setup. That is add `sudo` before each of the commands below.
* Build Base (torchaudio was difficult): `docker build -f docker/Dockerfile_base -t conv_ssl_base .`
* Build: `docker build . -t conv_ssl`
* Run: `docker run --rm -it --gpus all -v=$(pwd)/assets:/workspace/assets -v=$HOME/projects/data:/root/projects/data conv_ssl`
* Used during debug + some training:
  * Add current directory (if changing code)
  * Run: `docker run --rm -it --gpus all -v=$(pwd):/workspace -v=$HOME/projects/data:/root/projects/data conv_ssl`
  * Computational constraints
  * Run: `docker run --rm -it --gpus '"device=4,5,6,7"' -m=128g --cpus=16 --shm-size=16g -v=$(pwd):/workspace -v=$HOME/projects/data:/root/projects/data conv_ssl`

## Experiments

* Training uses [WandB](https://wandb.ai) by default.
* The event settings used in the paper are included in `conv_ssl/config/event_settings.json`.
  - See paper Section 3

```python
from conv_ssl.utils import read_json

event_settings = read_json("conv_ssl/config/event_settings.json")
hs_kwargs = event_settings['hs']
bc_kwargs = event_settings['bc']
metric_kwargs = event_settings['metric']
```

```json
{
  "hs": {
    "post_onset_shift": 1,
    "pre_offset_shift": 1,
    "post_onset_hold": 1,
    "pre_offset_hold": 1,
    "non_shift_horizon": 2,
    "metric_pad": 0.05,
    "metric_dur": 0.1,
    "metric_pre_label_dur": 0.5,
    "metric_onset_dur": 0.2
  },
  "bc": {
    "max_duration_frames": 1.0,
    "pre_silence_frames": 1.0,
    "post_silence_frames": 2.0,
    "min_duration_frames": 0.2,
    "metric_dur_frames": 0.2,
    "metric_pre_label_dur": 0.5
  },
  "metric": {
    "pad": 0.05,
    "dur": 0.1,
    "pre_label_dur": 0.5,
    "onset_dur": 0.2,
    "min_context": 3.0
  }
}
```

### Train

* Discrete
    ```bash
    python conv_ssl/train.py --gpus -1 --conf conv_ssl/config/model.yaml
    ```
* Independent: similar to [Skantze]() with bins like 'discrete'.
    ```bash
    python conv_ssl/train.py --gpus -1 --conf conv_ssl/config/model_independent.yaml
    ```
* Independent-40: similar to [Skantze]() with similar bin count.
    ```bash
    python conv_ssl/train.py --gpus -1 --conf conv_ssl/config/model_independent_baseline.yaml
    ```
* Comparative:
    ```bash
    python conv_ssl/train.py --gpus -1 --conf conv_ssl/config/comparative.yaml
    ```
* For faster training we saved all samples to disk and load directly
  - Save samples to disk: `conv_ssl/dataset_save_samples_to_disk.py` 
  - train on samples on disk: `conv_ssl/train_disk.py` 
  - SCRIPTS `conv_ssl/scripts`. Please look in script to get an idea of what they're doing...
    - `scripts/model_kfold.bash` for kfold training in paper
    - `scripts/test_models_script.bash` testing model training (`--fast_dev_run 1`)
    - `scripts/train_script.bash` testing model training (`--fast_dev_run 1`)


### Evaluate

```bash
python conv_ssl/evaluation/evaluation.py --checkpoint $PATH_TO_CHPT --savepath $PATH_TO_SAVEDIR --batch_size 4
```

### Paper

We ran model using kfold splits (see `conv_ssl/config/swb_kfolds`) over 4 different model architectures ('discrete', 'independent', 'independent-40', 'comparative').

* We evaluate (find threshold over validation set + final evaluation on test-set)
  - see `conv_ssl/evaluation/evaluate_paper_model.py`
  - the ids are the `WandB` ids.
  - We save all model scores to disk
* In `conv_ssl/evaluation/anova.py` we compare the scores to extract the final values in the paper.


## Citation

```latex
TBA
```

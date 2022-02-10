# Continuous Conversational SSL

* Causal
* Audio
* Timing


```bash
ConvSSL
├── assets
│   └── checkpoints
│       ├── cpc
│       │   └── 60k_epoch4-d0f474de.pt
│       ├── hubert
│       │   └── hubert_fairseq_base_ls960.pth
│       ├── wav2vec2
│       │   └── wav2vec2_fairseq_base_ls960.pth
│       └── wavlm
│           ├── WavLM-Base+.pt
│           └── WavLM-Base.pt
├── config
│   ├── conv_ssl.yaml
│   ├── ulm_small.yaml
│   └── ulm.yaml
├── tests/
├── encoder.py
├── projection_model.py
├── ulm_projection.py
├── dataset_kmean_idx.py
├── train_kmeans.py
├── evaluation.py
├── pretrained_encoders.py
├── utils.py
├── callbacks.py
└── README.md
```


## Installation

* Create conda env: `conda create -n conv_ssl python=3`
  - source env: `conda source conv_ssl`
* PyTorch: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
* [Optional] (for videos) Install FFMPEG: `conda install -c conda-forge ffmpeg`
* Dependencies: `pip install -r requirements.txt`
* install conv_ssl: `pip install -e .`
* Install [Datasets turn-taking](https://github.com/ErikEkstedt/datasets_turntaking)
    - clone repo `git clone https://github.com/ErikEkstedt/datasets_turntaking.git`
    - cd to repo, and install dependencies: `pip install -r requirements.txt`
    - install repo: `pip install -e .`

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

## Pretrained Encoders

The pretrained encoder checkpoints are downloaded from the original repos (see tree-structure above for paths)
or from torch.hub through torchaudio.

* [CPC](https://github.com/facebookresearch/CPC_audio)
  - requires installation of source
  - that is clone [CPC_audio](https://github.com/facebookresearch/CPC_audio), cd to repo, run `python setup.py develop`
  - **automatically** downloads checkpoints
* Unused
  * [WavLM-Base & WavLM-Base+](https://github.com/microsoft/unilm/tree/master/wavlm)
    - download the checkpoints
  * [Hubert & Wav2vec2](https://pytorch.org/audio/stable/models.html#wav2vec2-0-hubert) are downloaded through `torchaudio` (downloads to `~/.cache/torch/hub/checkpoints/` by default)
  * [Wav2vec, vq-wav2vec]()



## Experiments


Default

```bash
python conv_ssl/train.py --gpus -1 --conf conv_ssl/config/model.yaml
```

Regression with bins like default
```bash
python conv_ssl/train.py --gpus -1 --conf conv_ssl/config/model_regression.yaml
```

Regression with bins like [Skantze]()

```bash
python conv_ssl/train.py --gpus -1 --conf conv_ssl/config/model_regression_baseline.yaml
```


## UNUSED

```python
from torchaudio.pipelines import HUBERT_BASE, WAV2VEC2_BASE
#Hubert
model = HUBERT_BASE.get_model()
# or
model = WAV2VEC2_BASE.get_model()
```


## Train ULM Only (precompute units and save to disk)

### 1. Extract Features

Loads pretrained encoders (Hubert, wav2vec2) and extracts features over a subset of the
data (defined in memory (Gb) by `--kmeans_max_feature_memory`) which are (optionally) saved to disk.

Those features are then used to train a kmeans model (with k given by `--k`).

After the kmeans model is completed we iterate over the entire dataset (or
maximum batches, for debugging, given by `--feature_dataset_max_batches`) and
extracts the aligned VAD-labels and the Kmeans-indices which are saved to disk.

These features are then used to train a sequence model on the (fixed) units for any downstream task.


```bash
python conv_ssl/train_kmeans.py \
        --savepath dataset_units \
        --model hubert \
        --kmeans_max_feature_memory 10 \
        --extract_dataset # flag to also precompute units over the datasets
        # --overwrite_kmeans  # if you want to retrain kmeans
        # --overwrite_features  # if you want to re-extract features for kmeans
```


### 2. Train ULM + VAD-pred sequence models


```bash
python ulm_vad_prediction.py
```


### 3. Train SCPC+ model

```bash
python conv_ssl.py
```


## Model

We extend upon the [CPC]() model to encode hierarchical information relevant to
prosodic features (F0 and Activity). The model operates as the original CPC on
lower level features (100hz) while also reconstructing the F0, from the
observed data x, and predict a granular representation, of the future of the
conversation, which is defined the the Voice Activity over both channels.

Utilizing data from monologue recordings we can keep everything the same and
pretend like there is another interlocutor that is always quiet. We do not wish
to introduce duration information from the monologue speech such that the model
becomes bias to predict Holds.

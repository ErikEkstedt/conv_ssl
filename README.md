# Continuous Conversational SSL

* Causal
* Audio
* Timing


```bash
ConvSSL
├── config
│   ├── conv_ssl.yaml
│   ├── ulm_small.yaml
│   └── ulm.yaml
├── interaction
│   └── window_analysis.py
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
* Dependencies: `pip install -r requirements.txt`
* Install [Datasets turn-taking](https://github.com/ErikEkstedt/datasets_turntaking)
    - clone repo, cd to repo, and install dependencies: `pip install -r requirements.txt`
    - install repo: `pip install -e .`
* cd into this repo and install conv_ssl: `pip install -e .`


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

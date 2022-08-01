# Continuous Conversational SSL

Model training for the paper [Voice Activity Projection: Self-supervised Learning of Turn-taking Events](https://arxiv.org/abs/2205.09812).


## Installation

* Create conda env: `conda create -n conv_ssl python=3.9`
  - source env: `conda source conv_ssl`
* PyTorch: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
* Dependencies: 
  * Install requirements: `pip install -r requirements.txt`
  * **NOTE:** If you have problems install `pip install cython` manually first and then run the `pip install -r requirements.txt` command (trouble automating the install of the [CPC_audio](https://github.com/facebookresearch/CPC_audio) repo).
    * [Optional] Manual installation of [CPC_audio](https://github.com/facebookresearch/CPC_audio) (if the note above does not work)
      * `git clone https://github.com/facebookresearch/CPC_audio.git`
      * cd to repo and install dependencies (see repository) but probably you'll need
        * `pip install cython`
      * run: `python setup.py develop`  (again see original implementation)
  * **VAP**: Voice Activity Projection multi-purpose "head".
    * Install [`vap_turn_taking`](https://github.com/ErikEkstedt/vap_turn_taking)
      * `git clone https://github.com/ErikEkstedt/vap_turn_taking.git`
      * cd to repo, and install dependencies: `pip install -r requirements.txt`
      * Install: `pip install -e .`
  * **DATASET**
    * Install [datasets_turntaking](https://github.com/ErikEkstedt/datasets_turntaking)
      * `git clone https://github.com/ErikEkstedt/datasets_turntaking.git`
      * cd to repo, and install dependencies: `pip install -r requirements.txt`
      * Install repo: `pip install -e .`
    * **WARNING:** Requires [Switchboard](https://catalog.ldc.upenn.edu/LDC97S62) and/or [Fisher](https://catalog.ldc.upenn.edu/LDC2004S13) data!
* Install **`conv_ssl`:** 
  * cd to root directory and run: `pip install -e .`

### Train

```bash
python conv_ssl/train.py data.datasets=['switchboard','fisher'] +trainer.val_check_interval=0.5 early_stopping.patience=20
```

### Evaluate

```bash
python conv_ssl/evaluation/evaluation.py \
  +checkpoint_path=/full/path/checkpoint.ckpt \
  +savepath=assets/vap_fis \
  data.num_workers=4 \
  data.batch_size=16 
```


### Run

The `run.py` script loads a pretrained model and evaluates on a sample (waveform + `text_grid_name.TextGrid` or `vad_list_name.json`). See `examples` folder for format etc.

* Using defaults: `python run.py`
* Custom run requires a audio file `sample.wav` and **either** a `text_grid_name.TextGrid` or `vad_list_name.json`
  ```bash
  python run.py \
    -c example/cpc_48_50hz_15gqq5s5.ckpt \
    -w example/student_long_female_en-US-Wavenet-G.wav \ # waveform
    -tg example/student_long_female_en-US-Wavenet-G.TextGrid \ # text grid
    -v example/vad_list.json \ # vad-list
    -o VAP_OUTPUT.json  # output file
  ```


### Paper

The paper investigates the performance over kfold splits (see `conv_ssl/config/swb_kfolds`) over 4 different model architectures ('discrete', 'independent', 'independent-40', 'comparative').
* Save samples to disk: `conv_ssl/dataset_save_samples_to_disk.py` 
* train on samples on disk: `conv_ssl/train_disk.py` 
* run `scripts/model_kfold.bash`
* We evaluate (find threshold over validation set + final evaluation on test-set)
  - see `conv_ssl/evaluation/evaluate_paper_model.py`
  - the ids are the `WandB` ids.
  - We save all model scores to disk
* In `conv_ssl/evaluation/anova.py` we compare the scores to extract the final values in the paper.

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


## Citation

```latex
TBA
```

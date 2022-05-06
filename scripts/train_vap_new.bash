train="python conv_ssl/train_hydra.py +trainer.val_check_interval=0.5 early_stopping.patience=10"

data="data.datasets=['switchboard','fisher'] data.audio_duration=20 data.num_workers=24 data.batch_size=40"

dev="+trainer.limit_train_batches=10 +trainer.limit_val_batches=10"

# $train $data model=discrete_50hz $dev
$train $data model=discrete_50hz

# $train $data model=discrete_20hz $dev
$train  $data model=discrete_20hz

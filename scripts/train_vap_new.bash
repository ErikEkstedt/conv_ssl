train="python conv_ssl/train_hydra.py +trainer.val_check_interval=0.5"
data="data.datasets=['switchboard','fisher'] data.num_workers=24 data.batch_size=25"
dev="+trainer.limit_train_batches=10 +trainer.limit_val_batches=10"
dur20="data.audio_duration=20"
unfreeze10="optimizer.train_encoder_epoch=10"
unfreeze5="optimizer.train_encoder_epoch=5"
no_hist="data.vad_history=False"

#################################
# Train without History
#################################
# $train $data $dur20 $no_hist $unfreeze model=discrete_50hz model.ar.num_heads=8
# $train $data $dur20 $no_hist $unfreeze model=discrete_20hz model.ar.num_heads=8

#################################
# Train Encoder after 10 epochs
#################################
# $train $data $dur20 $unfreeze model=discrete_50hz model.ar.num_heads=8
# $train $data $dur20 $unfreeze model=discrete_20hz model.ar.num_heads=8


$train $data model=discrete model.ar.num_heads=8


$train $data $dur20 $unfreeze5 model=discrete_20hz model.ar.num_heads=8
$train $data $dur20 $unfreeze5 model=discrete_50hz model.ar.num_heads=8
$train $data $unfreeze5 model=discrete model.ar.num_heads=8

# $train $data model=discrete_20hz $dev
# $train $data $unfreeze model=discrete
# $train $data $no_hist $unfreeze model=discrete

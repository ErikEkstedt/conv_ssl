train="python conv_ssl/train_disk.py --gpus -1 --batch_size 50 --patience 50 --val_check_interval 0.5"

$train --conf conv_ssl/config/model.yaml
$train --conf conv_ssl/config/model_independent.yaml
$train --conf conv_ssl/config/model_independent_baseline.yaml
$train --conf conv_ssl/config/model_comparative.yaml

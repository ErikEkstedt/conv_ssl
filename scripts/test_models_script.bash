train="python conv_ssl/train.py --gpus -1 --batch_size 10 --fast_dev_run 1"

$train --conf conv_ssl/config/model.yaml
$train --conf conv_ssl/config/model_comparative.yaml
$train --conf conv_ssl/config/model_independent.yaml
$train --conf conv_ssl/config/model_independent_baseline.yaml

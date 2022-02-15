train="python conv_ssl/train.py --gpus -1 --batch_size 10 --fast_dev_run 1"

$train --conf conv_ssl/config/model.yaml
$train --conf conv_ssl/config/model_comparative.yaml
$train --conf conv_ssl/config/model_regression.yaml
$train --conf conv_ssl/config/model_regression_baseline.yaml

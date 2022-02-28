train="python conv_ssl/train_disk.py --gpus -1 --batch_size 10 --fast_dev_run 1"

$train --conf conv_ssl/config/model.yaml \
  --train_files conv_ssl/config/swb_kfolds/0_fold_train.txt \
  --val_files conv_ssl/config/swb_kfolds/0_fold_val.txt

$train --conf conv_ssl/config/model.yaml \
  --train_files conv_ssl/config/swb_kfolds/10_fold_train.txt \
  --val_files conv_ssl/config/swb_kfolds/10_fold_val.txt

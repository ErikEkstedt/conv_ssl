train="python conv_ssl/train_disk.py --gpus -1 --batch_size 50 --patience 20 --val_check_interval 0.5"


# Used in the paper to evaluate models over kfold-splits

for i in 0 1 2 3 4 5 6 7 8 9 10
do
 $train --conf conv_ssl/config/model.yaml \
    --train_files conv_ssl/config/swb_kfolds/${i}_fold_train.txt \
    --val_files conv_ssl/config/swb_kfolds/${i}_fold_val.txt \
    --seed $i \
    --name_info _kfold_${i}
done

for i in 0 1 2 3 4 5 6 7 8 9 10
do
 $train --conf conv_ssl/config/model_independent.yaml \
    --train_files conv_ssl/config/swb_kfolds/${i}_fold_train.txt \
    --val_files conv_ssl/config/swb_kfolds/${i}_fold_val.txt \
    --seed $i \
    --name_info _kfold_${i}
done

for i in 0 1 2 3 4 5 6 7 8 9 10
do
 $train --conf conv_ssl/config/model_independent_baseline.yaml \
    --train_files conv_ssl/config/swb_kfolds/${i}_fold_train.txt \
    --val_files conv_ssl/config/swb_kfolds/${i}_fold_val.txt \
    --seed $i \
    --name_info _kfold_${i}
done

for i in 0 1 2 3 4 5 6 7 8 9 10
do
 $train --conf conv_ssl/config/model_comparative.yaml \
    --train_files conv_ssl/config/swb_kfolds/${i}_fold_train.txt \
    --val_files conv_ssl/config/swb_kfolds/${i}_fold_val.txt \
    --seed $i \
    --name_info _kfold_${i}
done

# Equal bin-times for discrete
# for i in 0 1 2 3 4 5 6 7 8 9 10
# do
#  $train --conf conv_ssl/config/model_equal.yaml \
#     --train_files conv_ssl/config/swb_kfolds/${i}_fold_train.txt \
#     --val_files conv_ssl/config/swb_kfolds/${i}_fold_val.txt \
#     --seed $i \
#     --name_info _kfold_${i}
# done

# for i in 0 1 2 3 4 5 6 7 8 9 10
# do
#  $train --conf conv_ssl/config/model_latent.yaml \
#     --train_files conv_ssl/config/swb_kfolds/${i}_fold_train.txt \
#     --val_files conv_ssl/config/swb_kfolds/${i}_fold_val.txt \
#     --seed $i \
#     --name_info _kfold_${i}
# done

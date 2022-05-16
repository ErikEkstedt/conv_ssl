aug_eval="python conv_ssl/evaluation/evaluation_augmentation.py data.batch_size=4 data.num_workers=4"
swb="data.datasets=['switchboard']"
fisher="data.datasets=['fisher']"
both="data.datasets=['fisher', 'switchboard']"
paperb="/home/erik/projects/CCConv/conv_ssl/assets/PaperB"


# Paths
chpath="+checkpoint_path=/home/erik/projects/CCConv/conv_ssl/assets/PaperB/checkpoints/"
thpath="+threshold_path=/home/erik/projects/CCConv/conv_ssl/assets/PaperB/eval/"

# Model Checkpoints
cpc20=$chpath"cpc_48_20hz_2ucueis8.ckpt"
cpc50=$chpath"cpc_48_50hz_15gqq5s5.ckpt"
cpc100=$chpath"cpc_48_100hz_3mkvq5fk.ckpt"

# Model Thresholds
# th20_swb
# th20_fisher
# th20_both


sp_root="+savepath=PaperB/"
ch_root="+checkpoint_path="$paperb"/checkpoints/"
th_root="+threshold_path="$paperb"/eval/"
# cpc_44_20hz_unfreeze/thresholds.json


# savepaths
sp_root="+savepath=PaperB/"
$aug_eval $fisher $ch_root$cpc20u 


$aug_eval $swb $cpc20 $th20 +augmentation='low_pass' +cutoff_freq=250
$aug_eval $swb $cpc50 $th50 +augmentation='low_pass' +cutoff_freq=250
$aug_eval $swb $cpc100 $th100 +augmentation='low_pass' +cutoff_freq=250

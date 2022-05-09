aug_eval="python conv_ssl/evaluation/augmentation_eval.py data.batch_size=4 data.num_workers=4"
test_eval="python conv_ssl/evaluation/evaluation.py data.batch_size=10 data.num_workers=4"
swb="data.datasets=['switchboard']"
fisher="data.datasets=['fisher']"
both="data.datasets=['fisher', 'switchboard']"

# TODO: fix this scirpts

# TEST SAMPLE
# python conv_ssl/evaluation/evaluation.py 
# +checkpoint_path=/home/erik/projects/CCConv/conv_ssl/assets/PaperB/checkpoints/cpc_44_20hz_unfreeze_21ep_3btunf4b.ckpt 
# +savepath=assets/PaperB/cpc_44_20hz_unfreeze 
# data.num_workers=4 
# data.batch_size=10 
# data.datasets=['switchboard']

# AUGMENTATION SAMPLE
# python conv_ssl/evaluation/augmentation_eval.py 
# +checkpoint_path=/home/erik/projects/CCConv/conv_ssl/assets/PaperB/checkpoints/cpc_44_20hz_unfreeze_21ep_3btunf4b.ckpt 
# +savepath=PaperB/20hz_44_unfreeze 
# +threshold_path=/home/erik/projects/CCConv/conv_ssl/assets/PaperB/eval/cpc_44_20hz_unfreeze/thresholds.json 
# data.datasets=['switchboard'] 
# data.batch_size=4 
# data.num_workers=4



paperb="/home/erik/projects/CCConv/conv_ssl/assets/PaperB"
sp_root="+savepath=PaperB/"
ch_root="+checkpoint_path="$paperb"/checkpoints/"
th_root="+threshold_path="$paperb"/eval/"
# cpc_44_20hz_unfreeze/thresholds.json

# Checkpoints
# 20hz
cpc20u="cpc_44_20hz_unfreeze_21ep_3btunf4b.ckpt"
cpc20="cpc_48_20hz_2ucueis8.ckpt"
sp20=$sp_root"cpc_48_20"
sp20u=$sp_root"cpc_44_20u"

# 50hz
cpc50="cpc_48_50hz_15gqq5s5.ckpt"
sp50=$sp_root"cpc_48_50"
# 100hz
cpc100u="cpc_44_100hz_unfreeze_12ep.ckpt"
sp100=$sp_root"cpc_44_100"


# savepaths
sp_root="+savepath=PaperB/"

$test_eval $fisher $ch_root$cpc20u $sp20u
$aug_eval $fisher $ch_root$cpc20u 

# +checkpoint_path=/home/erik/projects/CCConv/conv_ssl/assets/PaperB/checkpoints/cpc_44_20hz_unfreeze_21ep_3btunf4b.ckpt 
# +savepath=assets/PaperB/cpc_44_20hz_unfreeze 



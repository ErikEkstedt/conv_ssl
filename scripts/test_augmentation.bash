aug_eval="python conv_ssl/evaluation/evaluation_augmentation.py data.batch_size=4 data.num_workers=4"
swb="data.datasets=['switchboard']"
fisher="data.datasets=['fisher']"
both="data.datasets=['fisher', 'switchboard']"

paperb="/home/erik/projects/CCConv/conv_ssl/assets/PaperB"


chpath="+checkpoint_path=/home/erik/projects/CCConv/conv_ssl/assets/PaperB/checkpoints/"

cpc20=$chpath"cpc_48_20hz_2ucueis8.ckpt"
cpc50=$chpath"cpc_48_50hz_15gqq5s5.ckpt"
cpc100=$chpath"cpc_48_100hz_3mkvq5fk.ckpt"

# DEBUG
# $aug_eval $swb $cpc50 +augmentation='flat_f0' +max_batches=10
# $aug_eval $swb $cpc50 +augmentation='shift_f0' +max_batches=10
# $aug_eval $swb $cpc50 +augmentation='flat_intensity' +max_batches=10
# $aug_eval $swb $cpc50 +augmentation='only_f0' +max_batches=10

# Augmentations: low_pass (only_f0), flat_f0, flat_intensity, equal_duration
$aug_eval $swb $cpc20 +augmentation='flat_f0'
$aug_eval $swb $cpc20 +augmentation='shift_f0'
$aug_eval $swb $cpc20 +augmentation='flat_intensity'
$aug_eval $swb $cpc20 +augmentation='only_f0'
$aug_eval $fisher $cpc20 +augmentation='flat_f0'
$aug_eval $fisher $cpc20 +augmentation='shift_f0'
$aug_eval $fisher $cpc20 +augmentation='flat_intensity'
$aug_eval $fisher $cpc20 +augmentation='only_f0'

$aug_eval $swb $cpc50 +augmentation='flat_f0'
$aug_eval $swb $cpc50 +augmentation='shift_f0'
$aug_eval $swb $cpc50 +augmentation='flat_intensity'
$aug_eval $swb $cpc50 +augmentation='only_f0'
$aug_eval $fisher $cpc50 +augmentation='flat_f0'
$aug_eval $fisher $cpc50 +augmentation='shift_f0'
$aug_eval $fisher $cpc50 +augmentation='flat_intensity'
$aug_eval $fisher $cpc50 +augmentation='only_f0'

$aug_eval $swb $cpc100 +augmentation='flat_f0'
$aug_eval $swb $cpc100 +augmentation='shift_f0'
$aug_eval $swb $cpc100 +augmentation='flat_intensity'
$aug_eval $swb $cpc100 +augmentation='only_f0'
$aug_eval $fisher $cpc100 +augmentation='flat_f0'
$aug_eval $fisher $cpc100 +augmentation='shift_f0'
$aug_eval $fisher $cpc100 +augmentation='flat_intensity'
$aug_eval $fisher $cpc100 +augmentation='only_f0'


# both last
$aug_eval $both $cpc20 +augmentation='flat_f0'
$aug_eval $both $cpc20 +augmentation='shift_f0'
$aug_eval $both $cpc20 +augmentation='flat_intensity'
$aug_eval $both $cpc20 +augmentation='only_f0'

$aug_eval $both $cpc50 +augmentation='flat_f0'
$aug_eval $both $cpc50 +augmentation='shift_f0'
$aug_eval $both $cpc50 +augmentation='flat_intensity'
$aug_eval $both $cpc50 +augmentation='only_f0'

$aug_eval $both $cpc100 +augmentation='flat_f0'
$aug_eval $both $cpc100 +augmentation='shift_f0'
$aug_eval $both $cpc100 +augmentation='flat_intensity'
$aug_eval $both $cpc100 +augmentation='only_f0'

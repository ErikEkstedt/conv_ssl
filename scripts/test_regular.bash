#!/bin/bash

# Evaluation of models.
# Using PR-curves and scores to find the best thresholds on validation set
# Use thresholds to get actual TEST-score

test_="python conv_ssl/evaluation/evaluation.py data.batch_size=10 data.num_workers=4"
swb="+data.datasets=['switchboard']"
fisher="+data.datasets=['fisher']"
both="+data.datasets=['fisher','switchboard']"
chpath="+checkpoint_path=/home/erik/projects/CCConv/conv_ssl/assets/PaperB/checkpoints/"

# regular
cpc20=$chpath"cpc_48_20hz_2ucueis8.ckpt"
cpc50=$chpath"cpc_48_50hz_15gqq5s5.ckpt"
cpc100=$chpath"cpc_48_100hz_3mkvq5fk.ckpt"

# 20 Hz
# $test_ $swb $cpc20
# $test_ $fisher $cpc20
# $test_ $both $cpc20

# 50 Hz
# $test_ $swb $cpc50
# $test_ $fisher $cpc50
# $test_ $both $cpc50

# 100 Hz
$test_ $swb $cpc100
$test_ $fisher $cpc100
$test_ $both $cpc100

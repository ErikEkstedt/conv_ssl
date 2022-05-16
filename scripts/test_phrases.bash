#!/bin/bash
phrase="python conv_ssl/evaluation/evaluation_phrases.py"

chpath="--checkpoint=/home/erik/projects/CCConv/conv_ssl/assets/PaperB/checkpoints/"
cpc20=$chpath"cpc_48_20hz_2ucueis8.ckpt"
cpc50=$chpath"cpc_48_50hz_15gqq5s5.ckpt"
cpc100=$chpath"cpc_48_100hz_3mkvq5fk.ckpt"

$phrase $cpc20
$phrase $cpc50
$phrase $cpc100

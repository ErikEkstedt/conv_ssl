#!/bin/bash

# 1. Install
# Create a new conda environment  (if you want), source environment and run:
# conda install -c conda-forge montreal-forced-aligner

# 2. Download acoustic-model/dictionary
# mfa model download acoustic english_us_arpa
# mfa model download dictionary english_us_arpa

# 3. validate corpus and align
corpus="assets/phrases_beta/duration_audio"
alignpath="assets/phrases_beta/duration_alignment"


# echo "Validating $corpus"
# mfa validate $corpus english_us_arpa english_us_arpa

echo "Alignment $corpus -> $alignpath"
mfa align $corpus english_us_arpa english_us_arpa $alignpath --clean

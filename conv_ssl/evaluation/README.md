# Evaluation


1. `python conv_ssl/evaluation/tts.py`
2. `python conv_ssl/evaluation/vad.py`
3. `python conv_ssl/evaluation/prepare_phrases_for_alignment.py`
  - puts .txt files with corresponding words suitable for montreal aligner
4. `bash conv_ssl/evaluation/forced_alignment.bash`
  - align files

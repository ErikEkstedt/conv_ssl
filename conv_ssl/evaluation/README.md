# Evaluation


1. Generate Audio: `python conv_ssl/evaluation/tts.py`
2. Aligner: 
  * `python conv_ssl/evaluation/prepare_phrases_for_alignment.py`
    - puts .txt files with corresponding words suitable for montreal aligner
  * `bash conv_ssl/evaluation/forced_alignment.bash`
    - align files (conda activate env with mfa montreal-forced-aligner)
4. Add VAD: `python conv_ssl/evaluation/vad.py`

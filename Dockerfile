# Conv_ssl
# Combine this with Dockerfile_base when everything is done and working...
FROM conv_ssl_base

COPY . .

# datasets_turntaking
WORKDIR /dependencies
RUN git clone https://github.com/ErikEkstedt/datasets_turntaking.git
WORKDIR /dependencies/datasets_turntaking
RUN pip install -r requirements.txt
RUN pip install -e .

# conv_ssl
WORKDIR /workspace
RUN pip install -r requirements.txt
RUN pip install -e .

# FairSeq (wav2vec)
WORKDIR /dependencies
RUN git clone https://github.com/pytorch/fairseq
WORKDIR /dependencies/fairseq
RUN pip install -e .

# back to workspace
WORKDIR /workspace
RUN pip install -U omegaconf  # fair reverts omegaconf

# Prepare switchboard (so we dont have to download it all the time)
RUN python docker/prepare_dataset.py

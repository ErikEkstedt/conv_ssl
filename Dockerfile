# Conv_ssl
# Combine this with Dockerfile_base when everything is done and working...
FROM conv_ssl_base

# datasets_turntaking
WORKDIR /dependencies
RUN git clone https://github.com/ErikEkstedt/datasets_turntaking.git
WORKDIR /dependencies/datasets_turntaking
RUN pip install -r requirements.txt
RUN pip install -e .

# vad_turn_taking
WORKDIR /dependencies
RUN git clone https://github.com/ErikEkstedt/vad_turn_taking.git
WORKDIR /dependencies/vad_turn_taking
RUN pip install -e .

# conv_ssl
WORKDIR /workspace
COPY . .
RUN pip install -r requirements.txt
RUN pip install -U omegaconf  # fair reverts omegaconf
RUN pip install -e .

# Prepare switchboard (so we dont have to download it all the time)
RUN python docker/prepare_dataset.py

# Conv_ssl
# Combine this with Dockerfile_base when everything is done and working...
FROM vap_base

# datasets_turntaking
WORKDIR /dependencies
RUN git clone https://github.com/ErikEkstedt/datasets_turntaking.git
WORKDIR /dependencies/datasets_turntaking
RUN pip install -r requirements.txt
RUN pip install -e .

# vad_turn_taking
WORKDIR /dependencies
RUN git clone https://github.com/ErikEkstedt/vap_turn_taking.git
WORKDIR /dependencies/vap_turn_taking
RUN pip install -r requirements.txt
RUN pip install -e .

# conv_ssl
WORKDIR /workspace
COPY . .
RUN pip install -r requirements.txt
RUN pip install -e .

# Prepare switchboard (so we dont have to download it all the time)
RUN python docker/prepare_dataset.py

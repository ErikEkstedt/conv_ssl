# Docker

**WARNING: this may not be updated (not using docker for development)**


* Requires [Nvidia-Docker]() for gpu support.
  * [Nvidia Docker Github](https://github.com/NVIDIA/nvidia-docker)
  * [github.io docs](https://nvidia.github.io/nvidia-docker/)
  * [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
* sudo may be required for default setup. That is add `sudo` before each of the commands below.
* Build Base (torchaudio was difficult): `docker build -f docker/Dockerfile_base -t vap_base .`
* Build: `docker build . -t vap`
* Run: `docker run --rm -it --gpus all -v=$(pwd)/assets:/workspace/assets -v=$HOME/projects/data:/root/projects/data vap`
* Used during debug + some training:
  * Add current directory (if changing code)
  * Run: `docker run --rm -it --gpus all -v=$(pwd):/workspace -v=$HOME/projects/data:/root/projects/data vap`


```bash
# takes time but installs torch/torchaudio etc + CPC model repository
docker build -f docker/Dockerfile_base -t vap_base .

# using the image above and installs VAP repos
# vap_turn_taking
# datasets_turntaking
# This repo
docker build . -t vap

# start docker (must include path to audio e.g. `$HOME/projects/data` in our case)
docker run --rm -it --gpus all -v=$(pwd)/assets:/workspace/assets -v=$HOME/projects/data:/root/projects/data vap
```


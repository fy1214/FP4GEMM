# base image
FROM nvcr.io/nvidia/pytorch:25.04-py3

# set env variable
ENV NVTE_FRAMEWORK=pytorch
ENV NVTE_WITH_USERBUFFERS=1
ENV MPI_HOME=/usr/local/mpi

# install
RUN apt-get update && apt-get install -y wget git

# install dependencies
RUN pip install sentencepiece
RUN pip install wandb

# install TransformerEngine
RUN git clone --branch release_v1.11 --recursive https://github.com/NVIDIA/TransformerEngine.git /workspace/TransformerEngine
WORKDIR /workspace/TransformerEngine
RUN pip wheel -w dist/ .

WORKDIR /workspace
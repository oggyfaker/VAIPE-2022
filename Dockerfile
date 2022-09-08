ARG CUDA_VERSION=11.3
ARG PYTHON_VERSION=3.9

FROM tiangolo/uvicorn-gunicorn:python3.9
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"


RUN apt update && apt-get install -y --no-install-recommends apt-utils && \
	apt-get install dialog -y && \
	apt-get -y install gcc && \
	apt install ffmpeg libsndfile1 --yes --no-install-recommends && \
    apt-get -y install git &&\
	apt-get install curl --yes --no-install-recommends &&\
    apt-get -y install wget --yes --no-install-recommends


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh


COPY requirements.txt ./
RUN conda create -y -n VAIPE2022 python=3.9
RUN /bin/bash -c "source activate VAIPE2022 \
                && pip install setuptools \
                && pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html\
                && pip install -r 'requirements.txt' "

#COPY ./ .


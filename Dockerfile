#FROM ajevnisek/tau-base-docker:latest
#RUN apt-get update && apt-get install -y \
#    python-pip
#RUN git clone https://github.com/orhir/latent-seg-diff.git
#WORKDIR latent-seg-diff/
#COPY environment.yaml .
#RUN conda update -y conda && \
#    conda env update --file environment.yaml
## Put conda in path
#RUN /bin/bash -c "source activate ldm"
#ENV CUDA_VISIBLE_DEVICES=1
#RUN pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
#RUN pip install git+https://github.com/arogozhnikov/einops.git
#RUN python main.py --base configs/autoencoder/autoencoder_kl_64x64x3_custom.yaml -t --gpus 0,1,2,3 --logdir /storage/orhir/stable_logs/
## WORKDIR ../
## RUN ./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_4.py 2



ARG UBUNTU_VER=16.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64


FROM nvidia/cuda:11.0.3-base-ubuntu${UBUNTU_VER}
RUN apt update
RUN apt install python3 python3-dev python3-pip wget build-essential git -y
RUN python3 -m pip install --upgrade pip
RUN apt-get clean -y
RUN rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

COPY . .
#WORKDIR /diffusion
RUN conda env create -f environment.yaml
# Make RUN commands use the new environment:
RUN echo "conda activate myenv" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
ENTRYPOINT ["./entrypoint.sh"]


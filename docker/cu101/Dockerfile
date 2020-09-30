FROM pytorch/pytorch:latest AS base
MAINTAINER Justin D Vrana "justin.vrana+caldera@gmail.com"
RUN pip install pip -U
RUN pip install poetry

FROM base AS torchscatter
ENV CUDA="cu101" TORCH="1.6.0"
RUN pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html

FROM torchscatter AS caldera
COPY . /src
WORKDIR /src
RUN pip install -r requirements.txt
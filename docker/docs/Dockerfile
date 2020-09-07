FROM python:3.8-buster AS base
ENV CUDA="cpu" TORCH="1.6.0"
MAINTAINER Justin D Vrana "justin.vrana+caldera@gmail.com"
RUN pip install pip -U
RUN pip install poetry

FROM base AS torch
RUN pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html

FROM torch AS torchscatter
RUN pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html

FROM torchscatter AS caldera
WORKDIR /src
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM caldera AS caldera-dev
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

FROM caldera AS caldera-docs
COPY requirements-docs.txt .
RUN pip install -r requirements-docs.txt

FROM caldera-docs AS project
COPY . .
RUN pip install .
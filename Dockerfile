FROM python:3.8-alpine AS base

MAINTAINER Justin D Vrana "justin.vrana+caldera@gmail.com"

RUN pip install pip -U

FROM base AS torch



ENV CUDA cu101
RUN pip install torch
RUN pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
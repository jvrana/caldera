import torch

print("Torch: {}".format(torch.__version__))
print("CUDA: {}".format(torch.cuda.is_available()))

import torch_scatter

print("torch_scatter: {}".format(torch_scatter))
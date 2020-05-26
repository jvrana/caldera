import torch
import torch_scatter

print("Torch: {}".format(torch.__version__))
print("CUDA: {}".format(torch.cuda.is_available()))

print("torch_scatter: {}".format(torch_scatter))

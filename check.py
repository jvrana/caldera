import torch

try:
    import torch_scatter
except ImportError as e:
    print(e)

print("Torch: {}".format(torch.__version__))
print("CUDA: {}".format(torch.cuda.is_available()))
print("torch_scatter: {}".format(torch_scatter))

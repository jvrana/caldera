import torch


class Select(torch.nn.Module):
    """Differentiable select block"""
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.blocks = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        out = self.blocks(latent)
        i = torch.where(torch.round(out) == 1)[0]
        return data[i]

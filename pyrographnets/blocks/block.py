from typing import Dict

from torch import nn


class Block(nn.Module):
    def __init__(self, module_dict: Dict[str, nn.Module], independent: bool):
        super().__init__()
        self._independent = independent
        self.block_dict = nn.ModuleDict(
            {name: mod for name, mod in module_dict.items() if mod is not None}
        )

    @property
    def independent(self):
        return self._independent

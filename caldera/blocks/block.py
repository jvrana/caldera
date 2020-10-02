from abc import ABC
from typing import Dict

from torch import nn


class Block(nn.Module, ABC):
    def __init__(self, module_dict: Dict[str, nn.Module], independent: bool):
        """A generic Graph Neural Network block.

        :param module_dict:
        :param independent:
        """
        super().__init__()
        self._independent = independent
        self.block_dict = nn.ModuleDict(
            {name: mod for name, mod in module_dict.items() if mod is not None}
        )

    @property
    def independent(self):
        return self._independent

    # def reset_parameters(self):
    #     for child in self.children():
    #         if hasattr(child, 'reset_parameters'):
    #             child.reset_parameters()

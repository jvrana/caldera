from torch import nn


class SequentialModuleDict(nn.ModuleDict):

    def forward(self, data):
        for v in self.values():
            data = v(data)
        return data
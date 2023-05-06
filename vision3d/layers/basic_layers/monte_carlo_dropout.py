import torch
import torch.nn as nn
import torch.nn.functional as F


class MonteCarloDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        out = F.dropout(x, p=self.p, training=True)
        return out


def convert_monte_carlo_dropout(module):
    module_output = module
    if isinstance(module, torch.nn.Dropout):
        module_output = MonteCarloDropout(module.p)
    for name, child in module.named_children():
        module_output.add_module(name, convert_monte_carlo_dropout(child))
    del module
    return module_output

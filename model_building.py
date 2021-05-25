import myfl as fnn
import torch
from torch import nn



def build_model(data_dimensions, num_hidden, num_blocks, mtype = 'realnvp', dev = None):
    num_inputs = data_dimensions
    num_cond_inputs = None

    if dev is None:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    modules = []

    if mtype == "maf":
        for _ in range(num_blocks):
            modules += [
                    fnn.MADE(num_inputs, num_hidden, None, act='tanh'),
                    fnn.BatchNormFlow(num_inputs),
                    fnn.Reverse(num_inputs)
                ]
    elif mtype == 'realnvp':
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(dev)
        for _ in range(num_blocks):
            modules += [
                fnn.CouplingLayer(
                    num_inputs, num_hidden, mask, None,
                    s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(num_inputs)
            ]
            mask = 1 - mask


    model = fnn.FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    return model

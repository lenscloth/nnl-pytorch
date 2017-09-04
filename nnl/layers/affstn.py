import torch as tr

from torch.nn import Module
from torch.nn import Parameter
from torch.nn.functional import grid_sample


class AffineSpatialTransform(Module):
    def __init__(self, OH, OW):
        super(AffineSpatialTransform, self).__init__()

        hs = tr.linspace(-1, 1, steps=OH)
        ws = tr.linspace(-1, 1, steps=OW)

        hs = hs.unsqueeze(1).expand(OH, OW)
        ws = ws.unsqueeze(0).expand(OH, OW)
        ones = tr.ones(OH, OW)

        t = tr.stack([ws, hs, ones], dim=2).unsqueeze(3)
        self.T = Parameter(t, requires_grad=False)

    def forward(self, input, theta):
        theta = theta.unsqueeze(1).unsqueeze(1)
        source_grid = tr.matmul(theta, self.T).squeeze(4)

        out = grid_sample(input, source_grid)

        return out

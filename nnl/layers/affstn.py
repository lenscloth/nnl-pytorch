import torch as tr

from torch.nn import Module
from torch.autograd import Variable
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
        self.register_buffer("t", t)

        vertices = tr.FloatTensor([[-1, -1, 1],
                                   [-1, 1, 1],
                                   [1, 1, 1],
                                   [1, -1, 1]]).unsqueeze(2)
        self.register_buffer("vertices", vertices)

    def forward(self, input, theta):
        theta = theta.unsqueeze(1).unsqueeze(1)
        t = Variable(self.t, requires_grad=False)
        v = Variable(self.vertices, volatile=True)

        source_grid = tr.matmul(theta, t).squeeze(4)
        out = grid_sample(input, source_grid)

        theta = theta.squeeze(1)
        vertices_out = tr.matmul(theta, v).squeeze(3)

        ih = input.size(2)
        iw = input.size(3)
        x_coord = (vertices_out[..., 0] + 1) * iw / 2
        y_coord = (vertices_out[..., 1] + 1) * ih / 2
        vertices_out = tr.stack((x_coord, y_coord), dim=2)

        return out, vertices_out

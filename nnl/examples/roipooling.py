import torch as tr

from torch.autograd import Variable
from torch.nn.modules.pooling import AdaptiveMaxPool2d
from nnl.layers.roipooling import RoiPooling

if __name__ == '__main__':
    input = Variable(tr.rand(1,1,10,10), requires_grad=True)
    rois = Variable(tr.LongTensor([[0,1,2,7,8],[0,3,3,8,8]]),requires_grad=False)

    out = AdaptiveMaxPool2d((3,3))(input)
    out.backward(out.data.clone().uniform_())

    pool = RoiPooling(size=(3, 3))
    out = pool(input, rois)
    out.backward(out.data.clone().uniform_())

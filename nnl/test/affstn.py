import torch as tr
import unittest

from torch.autograd import Variable
from nnl.layers.affstn import AffineSpatialTransform


class TestAffineSpatialTransform(unittest.TestCase):
    '''
    Identity matrix should produce same result
    '''
    def test_correct1(self):
        net = AffineSpatialTransform(10, 10)

        theta = Variable(tr.FloatTensor([[[1, 0, 0], [0, 1, 0]]]))
        input = Variable(tr.rand(1, 1, 10, 10))
        out, coord = net(input, theta)

        coord_ans = tr.FloatTensor([[0, 0], [0, 10], [10, 10], [10, 0]])
        self.assertEqual(tr.max(tr.abs(coord.data - coord_ans)), 0)
        self.assertAlmostEqual(tr.max(tr.abs(input - out).data), 0, places=5)

    '''
    Affine transformation 
      A = [2 0 0]
          [0 2 0]

      A-1 = [1/2 0  0]
            [0  1/2 0]
    '''
    def test_correct2(self):
        net = AffineSpatialTransform(10, 10)

        theta = Variable(tr.FloatTensor([[[1/2, 0, 0], [0, 1/2, 0]]]))
        input = Variable(tr.rand(1, 1, 20, 20))
        _, coord = net(input, theta)

        coord_ans = tr.FloatTensor([[5, 5], [5, 15], [15, 15], [15, 5]])

        self.assertEqual(tr.max(tr.abs(coord_ans - coord.data)), 0)

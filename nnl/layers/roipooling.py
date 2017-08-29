import torch as tr

from torch.nn.modules.pooling import AdaptiveMaxPool2d


class RoiPooling(tr.nn.Module):
    def __init__(self, size=(7, 7), spatial_scale=1.0):
        super(RoiPooling, self).__init__()

        self.size = size
        self.spatial_scale = spatial_scale

    def forward(self, feat, rois):
        assert (rois.dim() == 2)
        assert (rois.size(1) == 5)

        output = []
        rois = rois.data.float()
        num_rois = rois.size(0)

        rois[:, 1:].mul_(self.spatial_scale)
        rois = rois.long()
        for i in range(num_rois):
            roi = rois[i]
            im_idx = roi[0]
            im = feat[..., im_idx, roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
            output.append(AdaptiveMaxPool2d(self.size)(im))

        return tr.cat(output, 0)


from __future__ import print_function

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import math

from nnl.layers.affstn import AffineSpatialTransform
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image


batch_size = 100
learning_rate = 0.001
momentum = 0.9
epochs = 100
cuda = True

log_interval = 100
display = False


class RandomAffine:
    def __init__(self, seed):
        random.seed(seed)

    def __call__(self, image):
        cx, cy = image.size[0]/2, image.size[1]/2
        # scale matrix
        s1 = random.randrange(70, 120) / 100
        s2 = random.randrange(70, 120) / 100

        # rotation matrix
        angle = math.radians(random.randrange(0, 180))
        cos = math.cos(angle)
        sin = math.sin(angle)

        a1 = cos / s1
        a2 = sin / s1
        a3 = (-1 * sin) / s2
        a4 = cos / s2

        t1 = cx - (cx * a1) - (cy * a2) + random.randint(-10, 10)
        t2 = cy - (cx * a3) - (cy * a4) + random.randint(-10, 10)

        return image.transform(
            image.size,
            Image.AFFINE,
            (a1, a2, t1, a3, a4, t2),
            resample=Image.NEAREST)


train_loader = tr.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Pad(10),
                       RandomAffine(10),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=False)


test_loader = tr.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.Pad(10),
                       RandomAffine(10),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self, stn=False):
        super(Net, self).__init__()
        self.stn = stn

        self.loc_conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.loc_conv2 = nn.Conv2d(20, 20, kernel_size=5)
        self.loc_fc1 = nn.Linear(20 * 18 * 18, 20)
        self.loc_fc2 = nn.Linear(20, 6)
        self.loc_fc2.weight.data.fill_(0)
        self.loc_fc2.bias.data = tr.FloatTensor([1, 0, 0, 1, 0, 0])

        self.spatial = AffineSpatialTransform(48, 48)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(9 * 9 * 20, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        if self.stn:
            l = F.relu(F.max_pool2d(self.loc_conv1(x), 2))
            l = F.relu(self.loc_conv2(l))
            l = l.view(-1, 20 * 18 * 18)
            l = F.relu(self.loc_fc1(l))
            l = l.view(-1, 20)
            l = self.loc_fc2(l).view(-1, 2, 3)
            x = self.spatial(x, l)

        if display:
            img = x.data[0].transpose(0,2).squeeze().numpy()
            plt.imshow(img, cmap='gray')
            plt.show()
            print(l.data)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 9 * 9 * 20)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(model, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        if cuda:
            data = data.cuda()
            target = target.cuda()

        if display:
            img = data.data[0].transpose(0,2).squeeze().numpy()
            print(target.data[0])
            plt.imshow(img, cmap='gray')
            plt.show()

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if cuda:
            data = data.cuda()
            target = target.cuda()

        if display:
            img = data.data[0].transpose(0, 2).squeeze().numpy()
            print(target.data[0])
            plt.imshow(img, cmap='gray')
            plt.show()

        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    model = Net(stn=False)
    params_be_updated = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(params_be_updated, lr=learning_rate, momentum=momentum)

    if cuda:
        model.cuda()
    for epoch in range(1, epochs + 1):
        if epoch % 10 == 0:
            learning_rate = learning_rate / 2
            print("Update: learning rate %f" % learning_rate)
            params_be_updated = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.SGD(params_be_updated, lr=learning_rate, momentum=momentum)

        train(model, optimizer, epoch)
        test(model)

    display=True
    cuda=False

    model.cpu()
    test(model)
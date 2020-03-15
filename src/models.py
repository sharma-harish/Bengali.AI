import pretrainedmodels
import torch.nn as nn
class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = None)
        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)

    def forward(self, X):
        bs, _, _, _ = X.shape
        X = self.model.features(X)
        X = nn.functional.adaptive_avg_pool2d(X, 1).reshape(bs, -1)
        l0 = self.l0(X)
        l1 = self.l1(X)
        l2 = self.l2(X)
        return l0, l1, l2
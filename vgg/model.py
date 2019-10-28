import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import math
import os
import numpy as np
from helpers import id_to_name
import operator
import re


__all__ = [
    'VGG', 'vgg16',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes, latent_dim, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        ) 
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 1 * 1, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, num_classes)
        # )    
        if init_weights:
            self._initialize_weights()
        self.num_classes = num_classes

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '256': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    '128': [64, 64, 'M', 128, 128, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(num_classes, pretrained=False, latent_dim = 512, resume = False, path = False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    if latent_dim == 512:
        model = VGG(make_layers(cfg['D']), num_classes, latent_dim, **kwargs)
    if latent_dim == 256:
        model = VGG(make_layers(cfg['256']), num_classes, latent_dim, **kwargs)
    if latent_dim == 128:
        model = VGG(make_layers(cfg['128']), num_classes, latent_dim, **kwargs)
    

    if pretrained:        
        train_classes = os.listdir("C:/Users/peter/thesis/datasets/imagenet/train")
        train_classes.sort()
        #print(train_classes)
        train_classes = [re.sub("_"," ",class_) for class_ in train_classes]
        idx_name = [(idx,name.split(',')[0]) for (idx,name) in id_to_name.items() if name.split(',')[0] in train_classes]            
        idx_name.sort(key=operator.itemgetter(1))        
        idx = [int(idx_) for (idx_, name) in idx_name]
        state_dict = model_zoo.load_url(model_urls['vgg16'])
        state_dict['classifier.6.weight'] = state_dict['classifier.6.weight'][idx,:]
        state_dict['classifier.6.bias'] = state_dict['classifier.6.bias'][idx]
        model.load_state_dict((state_dict))
        
    elif resume:
        print("loading model from: ",path)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)

    return model
import json

import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, vgg_name, vgg_config, dropout=0.0):
        super(VGG, self).__init__()

        with open(vgg_config, "r") as f:
            cfg = json.load(f)

        self.input_size = 32
        self.features = self._make_layers(cfg[vgg_name])
        self.n_maps = cfg[vgg_name][-2]
        self.fc = self._make_fc_layers()
        self.classifier = nn.Linear(self.n_maps, 10)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, return_feat=False):
        out = self.features(x)
        if return_feat:
            return out.squeeze()
        out = out.view(out.size(0), -1)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.fc(out)
        out = self.classifier(out)
        return out

    def _make_fc_layers(self):
        layers = []
        layers += [
            nn.Linear(self.n_maps * self.input_size * self.input_size, self.n_maps),
            nn.BatchNorm1d(self.n_maps),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
                self.input_size = self.input_size // 2
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        return nn.Sequential(*layers)

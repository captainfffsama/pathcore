from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from utils import embedding_concat

class WideResnet502(nn.Module):
    def __init__(self):
        super(WideResnet502, self).__init__()
        self._model=models.wide_resnet50_2()
        self.conv1=self._model.conv1
        self.bn1=self._model.bn1
        self.relu=self._model.relu
        self.maxpool=self._model.maxpool

        self.layer1=self._model.layer1
        self.layer2=self._model.layer2
        self.layer3=self._model.layer3

        self.locally_aware_patch=torch.nn.AvgPool2d(3,1,1)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x1=self.layer2(x)
        x2=self.layer3(x1)

        x1=self.locally_aware_patch(x1)
        x2=self.locally_aware_patch(x2)
        
        z=embedding_concat(x1,x2)

        return z
    

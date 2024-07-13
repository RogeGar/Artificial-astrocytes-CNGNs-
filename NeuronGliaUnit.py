# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:00:42 2023

@author: Anonymous Author
"""

import torch
import torch.nn as nn


class NeuronGliaUnit(nn.Module):
    def __init__(self, layer, fort=1, dim=0.7, NG_pc=4, AS=2, Training:bool = True):
        super().__init__()
        self.layer = layer
        # if isinstance(self.layer,nn.Linear):
        #     self.layer_type = 'Linear'
        #     self.counters = torch.zeros(self.layer.out_features)
        if isinstance(self.layer,nn.Conv2d):
            self.layer_type = 'Conv 2d'
            self.counters = torch.zeros(self.layer.out_channels, requires_grad=False)
        self.NG_pc = NG_pc
        self.AS = AS
        self.fort = fort
        self.dim = dim
        self.Astro_count = 0

    def forward(self, x):

      x = self.layer(x)
      if self.training:
          with torch.no_grad():
            self.Astro_count += 1
            if self.layer_type == 'Linear':
                self.counters += torch.tanh(sum(x.clone())).detach().to('cpu')
            if self.layer_type == 'Conv 2d':
                self.counters += torch.max_pool2d(torch.tanh(sum(x.clone())), (torch.tanh(sum(x)).shape[1], torch.tanh(sum(x)).shape[1])).squeeze().detach().to('cpu')
    
            if self.Astro_count >= self.NG_pc :
              k2 = 0
              for param in list(self.layer.parameters())[0]:
                if self.counters[k2] > self.AS:
                  # print('S in action')
                  param = self.fort * param.clone()
                if self.counters[k2] < -self.AS:
                  # print('W in action')
                  param = self.dim * param.clone()
                k2 +=1
              self.Astro_count = 0
              self.counters *= 0
      return x

def transform_architecture(module, fort=1.5, dim=0.5, Astro=False):
  for id, child in module.named_children():
    if len(list(child.children())) > 0:
        if not isinstance(child,NeuronGliaUnit):
            Astro = True
        if isinstance(child,NeuronGliaUnit):
            child.fort = fort
            child.dim = dim
        transform_architecture(child, fort, dim, Astro)
    if len(list(child.children())) == 0:
      if isinstance(child,nn.Conv2d):
        if Astro:
          setattr(module, id, NeuronGliaUnit(child, fort, dim))
      # if isinstance(child,nn.Linear):
      #   setattr(module, id, NeuronGliaUnit(child))














# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:24:30 2023

@author: Anonymous Author
"""

import torch
import torch.nn as nn
import Hyperparameters
import NeuronGliaUnit
    
    
class RegNet(nn.Module):
    def __init__(self, model_key, classes, Astrocytes = False, Str = None, Weak = None, Pretrained = False, directory = None):
        super(RegNet, self).__init__()
        
        self.classes = classes
        self.model_key = model_key
        self.Astrocytes = Astrocytes
        self.Str = Str
        self.Weak = Weak
        self.Pretrained = Pretrained
        self.directory = directory
        self.RegNet = Hyperparameters.get_model(self.model_key, self.classes)
        if self.Astrocytes:
            NeuronGliaUnit.transform_architecture(self.RegNet, self.Str, self.Weak)
        if self.Pretrained:
            self.RegNet.load_state_dict(torch.load(directory,map_location='cuda:0'))
        self.data_counter = torch.zeros(len(classes))
        self.counters_in_zero = True
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.RegNet.stem(x)
        
        x = self.RegNet.trunk_output.block1(x)
        self.Activations1 = x.clone().detach()
        x = self.RegNet.trunk_output.block2(x)
        self.Activations2 = x.clone().detach()
        x = self.RegNet.trunk_output.block3(x)
        self.Activations3 = x.clone().detach()
        x = self.RegNet.trunk_output.block4(x)
        self.Activations4 = x.clone().detach()
        if  x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        x = self.RegNet.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.RegNet.fc(x)
            
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, pooled=True):
        self.a1 = self.Activations1
        self.a2 = self.Activations2
        self.a3 = self.Activations3
        self.a4 = self.Activations4
        
        if pooled:
            self.a1 = Global_avg_2dpooling(self.Activations1)
            self.a2 = Global_avg_2dpooling(self.Activations2)
            self.a3 = Global_avg_2dpooling(self.Activations3)
            self.a4 = Global_avg_2dpooling(self.Activations4)
        
        return self.a1, self.a2, self.a3, self.a4

class MobileNetV3(nn.Module):
    def __init__(self, model_key, classes, Astrocytes = False, Str = None, Weak = None, Pretrained = False, directory = None):
        super(MobileNetV3, self).__init__()
        
        self.classes = classes
        self.model_key = model_key
        self.Astrocytes = Astrocytes
        self.Str = Str
        self.Weak = Weak
        self.Pretrained = Pretrained
        self.directory = directory
        self.MobileNetV3 = Hyperparameters.get_model(self.model_key, self.classes)
        if self.Astrocytes:
           NeuronGliaUnit.transform_architecture(self.MobileNetV3, self.Str, self.Weak)
        if self.Pretrained:
           self.MobileNetV3.load_state_dict(torch.load(directory,map_location='cuda:0'))
        self.data_counter = torch.zeros(len(classes))
        self.counters_in_zero = True
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.MobileNetV3.features(x)
        
        self.Activations1 = None
        self.Activations2 = None
        self.Activations3 = None
        self.Activations4 = x.clone().detach()
        if  x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        x = self.MobileNetV3.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.MobileNetV3.classifier(x)
            
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, pooled=True):
        # self.a1 = self.Activations1
        # self.a2 = self.Activations2
        # self.a3 = self.Activations3
        self.a4 = self.Activations4
        
        if pooled:
            # self.a1 = Global_avg_2dpooling(self.Activations1)
            # self.a2 = Global_avg_2dpooling(self.Activations2)
            # self.a3 = Global_avg_2dpooling(self.Activations3)
            self.a4 = Global_avg_2dpooling(self.Activations4)
        
        return torch.empty(1), torch.empty(1), torch.empty(1), self.a4

class ShufflenetV2(nn.Module):
    def __init__(self, model_key, classes, Astrocytes = False, Str = None, Weak = None, Pretrained = False, directory = None):
        super(ShufflenetV2, self).__init__()
        
        self.classes = classes
        self.model_key = model_key
        self.Astrocytes = Astrocytes
        self.Str = Str
        self.Weak = Weak
        self.Pretrained = Pretrained
        self.directory = directory
        self.ShufflenetV2 = Hyperparameters.get_model(self.model_key, self.classes)
        if self.Astrocytes:
           NeuronGliaUnit.transform_architecture(self.ShufflenetV2, self.Str, self.Weak)
        if self.Pretrained:
           self.ShufflenetV2.load_state_dict(torch.load(directory,map_location='cuda:0'))
        self.data_counter = torch.zeros(len(classes))
        self.counters_in_zero = True
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.ShufflenetV2.conv1(x)
        
        x = self.ShufflenetV2.maxpool(x)
        self.Activations1 = x.clone().detach()
        x = self.ShufflenetV2.stage2(x)
        self.Activations2 = x.clone().detach()
        x = self.ShufflenetV2.stage3(x)
        self.Activations3 = x.clone().detach()
        x = self.ShufflenetV2.stage4(x)
        self.Activations4 = x.clone().detach()
        if  x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        x = self.ShufflenetV2.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.ShufflenetV2.fc(x)
            
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, pooled=True):
        self.a1 = self.Activations1
        self.a2 = self.Activations2
        self.a3 = self.Activations3
        self.a4 = self.Activations4
        
        if pooled:
            self.a1 = Global_avg_2dpooling(self.Activations1)
            self.a2 = Global_avg_2dpooling(self.Activations2)
            self.a3 = Global_avg_2dpooling(self.Activations3)
            self.a4 = Global_avg_2dpooling(self.Activations4)
        
        return self.a1, self.a2, self.a3, self.a4
    
class EfficientNetV2(nn.Module):
    def __init__(self, model_key, classes):
        super(EfficientNetV2, self).__init__()
        
        self.classes = classes
        self.model_key = model_key
        self.EfficientNetV2 = Hyperparameters.get_model(self.model_key, self.classes)
        self.data_counter = torch.zeros(len(classes))
        self.counters_in_zero = True
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.EfficientNetV2.features[0](x)
        x = self.EfficientNetV2.features[1](x)
        self.Activations1 = x.clone().detach()
        x = self.EfficientNetV2.features[2](x)
        x = self.EfficientNetV2.features[3](x)
        self.Activations2 = x.clone().detach()
        x = self.EfficientNetV2.features[4](x)
        x = self.EfficientNetV2.features[5](x)
        self.Activations3 = x.clone().detach()
        x = self.EfficientNetV2.features[6](x)
        x = self.EfficientNetV2.features[7](x)
        self.Activations4 = x.clone().detach()
        if  x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        x = self.EfficientNetV2.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.EfficientNetV2.classifier(x)
            
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, pooled=True):
        self.a1 = self.Activations1
        self.a2 = self.Activations2
        self.a3 = self.Activations3
        self.a4 = self.Activations4
        
        if pooled:
            self.a1 = Global_avg_2dpooling(self.Activations1)
            self.a2 = Global_avg_2dpooling(self.Activations2)
            self.a3 = Global_avg_2dpooling(self.Activations3)
            self.a4 = Global_avg_2dpooling(self.Activations4)
        
        return self.a1, self.a2, self.a3, self.a4
    
   
    
def Global_avg_2dpooling(tensor):
    pooled_tensor = torch.nn.functional.avg_pool2d(tensor, (tensor.shape[-1], tensor.shape[-1])).squeeze().detach()
    
    return pooled_tensor
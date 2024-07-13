# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:11:35 2023

@author: Anonymous Author
"""

import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

All_Available_models = ['ResNet-50', 'ViT-B-16', 'ConvNext-Tiny', 'ConvNext-Small',
                    'EfficientNet-V2-Small', 'RegNet-Y-400MF', 'RegNet-X-400MF',
                    'MobileNet-V3-Small', 'MobileNet-V3-Large', 'SwinTranfsormer-Tiny',
                    'SwinTranfsormer-Small', 'SwinTranfsormer-Base'
                    'Shufflenet-v2-x1_0', 'Shufflenet-v2-x1_5', 'Shufflenet-v2-x2_0'
                    ]

Mobile_Available_models = ['ConvNext-Tiny', 'EfficientNet-V2-Small', 'RegNet-Y-400MF', 'RegNet-X-400MF',
                    'MobileNet-V3-Small', 'MobileNet-V3-Large', 'Shufflenet-v2-x1_0', 
                    'Shufflenet-v2-x1_5', 'Shufflenet-v2-x2_0']

Available_losses = ['cross-entropy', 'label smoothing', 'logit penalty',
                    'logit normalization', 'sigmoid cross-entropy']

Available_optimizers = ['Adam', 'SGD', 'Adamax', 'RMSprop']

def get_model(kind_model, classes):
    
    
    if kind_model == 'ResNet-50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(2048, len(classes))
        
    if kind_model == 'ViT-B-16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(768, len(classes))
        """
        In accordance with the original ViT paper:
            
            'When transferring ViT models to another dataset, we remove 
            the whole head (two linear layers) and replace it by a single, 
            zero-initialized linear layer outputting the number of classes 
            required by the target dataset. We found this to be a little 
            more robust than simply re-initializing the very last layer.'
                            
                            Dosovitskiy et. al., An Image is Worth 
                            16x16 Words: Transformers for Image Recognition 
                            at Scale, https://arxiv.org/abs/2010.11929
        
        """
    if kind_model == 'ConvNext-Tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(768, len(classes))
        
    if kind_model == 'ConvNext-Small':
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(768, len(classes))
        
    if kind_model == 'EfficientNet-V2-Small':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(1280, len(classes))
        
    if kind_model == 'RegNet-Y-400MF':
        model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
        model.fc = nn.Linear(440, len(classes))
        
    if kind_model == 'RegNet-Y-800MF':
        model = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.DEFAULT)
        model.fc = nn.Linear(784, len(classes))
        
    if kind_model == 'RegNet-Y-1_6GF':
        model = models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.DEFAULT)
        model.fc = nn.Linear(888, len(classes))
        
    if kind_model == 'RegNet-X-400MF':
        model = models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.DEFAULT)
        model.fc = nn.Linear(400, len(classes))
        
    if kind_model == 'RegNet-X-800MF':
        model = models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.DEFAULT)
        model.fc = nn.Linear(672, len(classes))
        
    if kind_model == 'RegNet-X-1_6GF':
        model = models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.DEFAULT)
        model.fc = nn.Linear(912, len(classes))
        
    if kind_model == 'MobileNet-V3-Small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(1024, len(classes))
        
    if kind_model == 'MobileNet-V3-Large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(1280, len(classes))
        
    if kind_model == 'SwinTranfsormer-Tiny':
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(768, len(classes))
        
    if kind_model == 'SwinTranfsormer-Small':
        model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        model.head = nn.Linear(768, len(classes))
    
    if kind_model == 'SwinTranfsormer-Base':
        model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        model.head = nn.Linear(1024, len(classes))
        
    if kind_model == 'Shufflenet-v2-x1_0':
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        model.fc = nn.Linear(1024, len(classes))
        
    if kind_model == 'Shufflenet-v2-x1_5':
        model = models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.DEFAULT)
        model.fc = nn.Linear(1024, len(classes))
    
    if kind_model == 'Shufflenet-v2-x2_0':
        model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT)
        model.fc = nn.Linear(2048, len(classes))
        
    return model

def get_loss(loss_fun, preds, targets, alpha=0.1, beta=0.0004, tau=0.04):
    I,K = preds.shape
    loss=torch.tensor(0)
    if loss_fun == 'cross-entropy':
        for i in range(I):
            loss = loss - torch.dot(targets[i],preds[i]) + torch.log(sum(torch.exp(preds[i]))) 
    
    elif loss_fun == 'label smoothing':
        for i in range(I):
            loss = loss - torch.dot(targets[i],preds[i]) + (1/(1-alpha))*torch.log(sum(torch.exp(preds[i]))) - (alpha/((1-alpha)*K))*sum(preds[i])
        
    elif loss_fun == 'logit penalty':
        for i in range(I):
            loss = loss - torch.dot(targets[i],preds[i]) + torch.log(sum(torch.exp(preds[i]))) +  beta*torch.square(torch.norm(preds[0],2))
    
    elif loss_fun == 'logit normalization':
        for i in range(I):
            lognorm = torch.div(preds[i], tau*torch.norm(preds[i]))
            loss = loss - torch.dot(targets[i],lognorm) + torch.log(sum(torch.exp(lognorm))) 
            
    else: #loss_fun == 'sigmoid cross-entropy'
        for i in range(I):
            loss = loss - torch.dot(targets[i],preds[i])  + sum(torch.log(torch.exp(preds[i]) + 1))
    
    return torch.div(loss,I)

def get_optimizer(optim_fun, network_parameters, lr_op, weight_decay):
    if optim_fun == 'Adam':
        optimizer = optim.Adam(network_parameters, lr=0.001*lr_op)
    
    elif optim_fun == 'SGD':
        optimizer = optim.SGD(network_parameters, lr=0.01*lr_op, momentum=0.9, weight_decay=weight_decay)
        
    elif optim_fun == 'RMSprop':
        optimizer = optim.RMSprop(network_parameters, lr=0.001*lr_op, weight_decay=weight_decay, momentum=0.9)
            
    else: #optim_fun == 'Adamax'
        optimizer = optim.Adamax(network_parameters, lr=0.002*lr_op, weight_decay=weight_decay)
    
    return optimizer
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:25:35 2024

@author: Rogelio Garcia
"""

import torch
import torchvision
import numpy as np
import pandas as pd
import Dataset_creator
import NeuronGliaUnit
import Performance_metrics_CNGN
import time
import Hyperparameters
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2 as T
import torchvision.transforms as transforms



def training_loop_CNGN(model_key, optimizer_key, lr, weight_decay, loss_key, train_loader, valid_loader, classes, epochs, 
                  device = 'cpu', Training_time = False, scheduler = False, Astro=False, Stre = 1, Weak = 1, Prev = False, 
                  prev_eps_ep = 0, prev_eps_it = 0, identifier=None, cutmix_mixup = False, tr_metrics = False, vl_metrics = False):
    model = Hyperparameters.get_model(model_key, classes)
    metrics_tr_data = []
    metrics_vl_data = []
    D = False
    if Astro:
        print('Transforming to CNGN')
        if type(Stre) is int or type(Stre) is float:
            NeuronGliaUnit.transform_architecture(model, Stre, Weak)
        else:
            D = True
    
    if Prev:
        model.load_state_dict(torch.load(identifier))
    
    model.to(device)
    BT = time.time()
    optimizer = Hyperparameters.get_optimizer(optimizer_key, model.parameters(), lr, weight_decay)
    
    if cutmix_mixup:
        cutmix = T.CutMix(num_classes=len(classes))
        mixup = T.MixUp(num_classes=len(classes))
        cutmix_or_mixup = T.RandomChoice([cutmix, mixup])
    
    if scheduler:
        it_steps = len(train_loader)
        Scheduler = CosineAnnealingLR(optimizer,
                                      T_max = int(it_steps*epochs), # Maximum number of iterations.
                                     eta_min = optimizer.param_groups[0]['lr']*0.1) # Minimum learning rate.
        if Prev:
            for Sced_dumi in range(prev_eps_it*it_steps*epochs):
                Scheduler.step()
    
    for epoch in range(epochs-prev_eps_ep):
        model.train()
        total_loss = 0
        if D:
            NeuronGliaUnit.transform_architecture(model, Stre[epoch], Weak[epoch])
        for batch in train_loader:
            torch.cuda.empty_cache()
            images = batch[0].to(device)
            labels = batch[1].to(device)
            batch[0], batch[1] = cutmix_or_mixup(batch[0], batch[1])
            preds = model(images)
            loss = Hyperparameters.get_loss(loss_key, preds, F.one_hot(labels, len(classes)).float())
            del images
            del labels
            del preds
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            del loss
            optimizer.step()
            if scheduler:
                Scheduler.step()
                
                
        AT = time.time()
        if Training_time:
            print('Epoch: ', epoch + prev_eps_ep, 'Time: ', format((AT-BT)/60, '.2f'), ' minutes')
        
        if tr_metrics:
            model.eval()
            preds_tr, labels_tr, _= Performance_metrics_CNGN.get_all_preds(model, train_loader, device, get_activations = False)
            metrics_tr, _ = Performance_metrics_CNGN.get_performance_metrics(classes, labels_tr, preds_tr)
            metrics_tr_data.append([metrics_tr.loc['cat-acc'][0], metrics_tr.loc['micro-f1'][0], metrics_tr.loc['mcc'][0], total_loss])
            print('Training Cat-ACC: ', metrics_tr.loc['cat-acc'][0])
            del preds_tr, labels_tr, metrics_tr
        
        if vl_metrics:
            model.eval()
            preds_vl, labels_vl, _= Performance_metrics_CNGN.get_all_preds(model, valid_loader, device, get_activations = False)
            print('Val preds computed, estimating metrics...')
            metrics_vl, _ = Performance_metrics_CNGN.get_performance_metrics(classes, labels_vl, preds_vl)
            metrics_vl_data.append([metrics_vl.loc['cat-acc'][0], metrics_vl.loc['micro-f1'][0], metrics_vl.loc['mcc'][0]])
            print('Validation Cat-ACC: ', metrics_vl.loc['cat-acc'][0])
            del preds_vl, labels_vl, metrics_vl
        
    return model, metrics_tr_data, metrics_vl_data


def CIFAR10_splits(size, batch=32, seed=0, split = 0.9):
    directory = 'C:/Users/Rogelio Garcia/Documents/Doctorado/Images datasets'
    g = torch.Generator()
    g.manual_seed(seed)

    # Reproducibility *******************************************************************************

    train_crop = size
    train_size = round(train_crop*1.1)
    val_size = size
    test_size = size

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    Transforms = transforms.Compose([
                    #transforms.ToPILImage(),
                    transforms.Resize(train_size),
                    transforms.RandomCrop(size=train_crop),
                    transforms.RandomRotation(degrees=(90)),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomHorizontalFlip(0.5),
                    torchvision.transforms.ColorJitter(brightness=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
    Transforms_test = transforms.Compose([
                #transforms.ToPILImage(),
                transforms.Resize(test_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])

    TR_set = torchvision.datasets.CIFAR10(directory, train = True, transform = None, target_transform = None, download = True)
    test_set = torchvision.datasets.CIFAR10(directory, train = False, transform = Transforms_test, target_transform = None, download = False)
    train_set, valid_set = torch.utils.data.random_split(TR_set,(round(len(TR_set)*split), len(TR_set) - round(len(TR_set)*split)), generator = g)
    classes = TR_set.classes
    del TR_set
    train_set = Dataset_creator.DummiDataset(train_set, transform=Transforms)
    valid_set = Dataset_creator.DummiDataset(valid_set, transform=Transforms_test)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, num_workers=1,
                                                        shuffle=True,
                                                        # worker_init_fn=seed,
                                                        # generator=g,
                                                        )
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch, num_workers=1,
                                                  shuffle=False,
                                                  # worker_init_fn=seed,
                                                  # generator=g,
                                                  )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, num_workers=1,
                                                  shuffle=False,
                                                  # worker_init_fn=seed,
                                                  # generator=g,
                                                  )
    
    return train_loader, valid_loader, test_loader, classes

def CIFAR100_splits(size, batch=32, seed=0, split = 0.9):
    directory = 'C:/Users/Rogelio Garcia/Documents/Doctorado/Images datasets'
    g = torch.Generator()
    g.manual_seed(seed)

    # Reproducibility *******************************************************************************

    train_crop = size
    train_size = round(train_crop*1.1)
    val_size = size
    test_size = size

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    Transforms = transforms.Compose([
                    #transforms.ToPILImage(),
                    transforms.Resize(train_size),
                    transforms.RandomCrop(size=train_crop),
                    transforms.RandomRotation(degrees=(90)),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomHorizontalFlip(0.5),
                    torchvision.transforms.ColorJitter(brightness=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
    Transforms_test = transforms.Compose([
                #transforms.ToPILImage(),
                transforms.Resize(test_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])

    TR_set = torchvision.datasets.CIFAR100(directory, train = True, transform = None, target_transform = None, download = True)
    test_set = torchvision.datasets.CIFAR100(directory, train = False, transform = Transforms_test, target_transform = None, download = False)
    train_set, valid_set = torch.utils.data.random_split(TR_set,(round(len(TR_set)*split), len(TR_set) - round(len(TR_set)*split)), generator = g)
    classes = TR_set.classes
    del TR_set
    train_set = Dataset_creator.DummiDataset(train_set, transform=Transforms)
    valid_set = Dataset_creator.DummiDataset(valid_set, transform=Transforms_test)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, num_workers=1,
                                                        shuffle=True,
                                                        # worker_init_fn=seed,
                                                        # generator=g,
                                                        )
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch, num_workers=1,
                                                  shuffle=False,
                                                  # worker_init_fn=seed,
                                                  # generator=g,
                                                  )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, num_workers=1,
                                                  shuffle=False,
                                                  # worker_init_fn=seed,
                                                  # generator=g,
                                                  )
    
    return train_loader, valid_loader, test_loader, classes

def get_confs(Model_key = 'Shufflenet-v2-x1_5', DATABASE = 'CIFAR-10',
              directory = 'C:/Users/Rogelio Garcia/Documents/Doctorado/8 semestre/Results astrocytes on RIT/',
              repetitions = 3, epochs = [0, 1, 2, 3, 4], astr_values = [0.5, 0.75, 1, 1.25, 1.5]):
    
    
    Exp_dir = directory + DATABASE + '/Results/' + Model_key + '/' 
            
    # if os.path.isdir(Exp_dir) is False:
    #     os.makedirs(Exp_dir)
    
    
    tr_acc = np.zeros((len(astr_values),len(astr_values),len(epochs),repetitions))
    vl_acc = np.zeros((len(astr_values),len(astr_values),len(epochs),repetitions))
    
    tr_f1 = np.zeros((len(astr_values),len(astr_values),len(epochs),repetitions))
    vl_f1 = np.zeros((len(astr_values),len(astr_values),len(epochs),repetitions))
    
    tr_mcc = np.zeros((len(astr_values),len(astr_values),len(epochs),repetitions))
    vl_mcc = np.zeros((len(astr_values),len(astr_values),len(epochs),repetitions))
    
    tr_loss = np.zeros((len(astr_values),len(astr_values),len(epochs),repetitions))
    vl_loss = np.zeros((len(astr_values),len(astr_values),len(epochs),repetitions))
    
    for rep in range(repetitions):
        
        for i in range(len(astr_values)):
            for j in range(len(astr_values)): 
                
                
                d_tr_acc = pd.read_csv(Exp_dir + Model_key + '_acc_tr_' + str(astr_values[i]) + '_' + str(astr_values[j]) + '_' + 'rep' + str(rep) + '.csv').to_numpy()
                d_tr_f1 = pd.read_csv(Exp_dir + Model_key + '_f1_tr_' + str(astr_values[i]) + '_' + str(astr_values[j]) + '_' + 'rep' + str(rep) + '.csv').to_numpy()
                d_tr_mcc = pd.read_csv(Exp_dir + Model_key + '_mcc_tr_' + str(astr_values[i]) + '_' + str(astr_values[j]) + '_' + 'rep' + str(rep) + '.csv').to_numpy()
                d_tr_loss = pd.read_csv(Exp_dir + Model_key + '_loss_tr_' + str(astr_values[i]) + '_' + str(astr_values[j]) + '_' + 'rep' + str(rep) + '.csv').to_numpy()
                
                d_vl_acc = pd.read_csv(Exp_dir + Model_key + '_acc_val_' + str(astr_values[i]) + '_' + str(astr_values[j]) + '_' + 'rep' + str(rep) + '.csv').to_numpy()
                d_vl_f1 = pd.read_csv(Exp_dir + Model_key + '_f1_val_' + str(astr_values[i]) + '_' + str(astr_values[j]) + '_' + 'rep' + str(rep) + '.csv').to_numpy()
                d_vl_mcc = pd.read_csv(Exp_dir + Model_key + '_mcc_val_' + str(astr_values[i]) + '_' + str(astr_values[j]) + '_' + 'rep' + str(rep) + '.csv').to_numpy()
                d_vl_loss = pd.read_csv(Exp_dir + Model_key + '_loss_val_' + str(astr_values[i]) + '_' + str(astr_values[j]) + '_' + 'rep' + str(rep) + '.csv').to_numpy()
                
                
                for epoch in range(len(epochs)):
                    
                    tr_acc[i,j,epoch,rep] = d_tr_acc[epoch,1]
                    tr_f1[i,j,epoch,rep] = d_tr_f1[epoch,1]
                    tr_mcc[i,j,epoch,rep] = d_tr_mcc[epoch,1]
                    tr_loss[i,j,epoch,rep] = d_tr_loss[epoch,1]
                    
                    vl_acc[i,j,epoch,rep] = d_vl_acc[epoch,1]
                    vl_f1[i,j,epoch,rep] = d_vl_f1[epoch,1]
                    vl_mcc[i,j,epoch,rep] = d_vl_mcc[epoch,1]
                    vl_loss[i,j,epoch,rep] = d_vl_loss[epoch,1]
    
    
    
    
                    
                    
    tr_acc_m = np.mean(tr_acc, axis=3)
    tr_f1_m = np.mean(tr_f1, axis=3)
    tr_mcc_m = np.mean(tr_mcc, axis=3)
    tr_loss_m = np.mean(tr_loss, axis=3)
    
    vl_acc_m = np.mean(vl_acc, axis=3)
    vl_f1_m = np.mean(vl_f1, axis=3)
    vl_mcc_m = np.mean(vl_mcc, axis=3)
    vl_loss_m = np.mean(vl_loss, axis=3)
    
    confs = []
    
    for epoch in range(len(epochs)):
        ind = np.unravel_index(np.argmax(vl_acc_m[:,:,epoch], axis=None), vl_acc_m[:,:,epoch].shape)
        confs.append([astr_values[ind[0]],astr_values[ind[1]]])
    return confs

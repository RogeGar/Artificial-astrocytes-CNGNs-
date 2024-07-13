# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:56:47 2023

@author: Anonymous Author
"""

import numpy as np
import torch
import pandas as pd
import Dataset_creator
import Hyperparameters

def get_all_preds(model, loader, device, get_activations = False):
    model.eval()
    activations = 'Activations were not required'
    first = True
    with torch.no_grad():
        all_preds = torch.tensor([]).to(device)
        true_preds = torch.tensor([]).to(device)
        for batch in loader:
            torch.cuda.empty_cache()
            images = batch[0].to(device) 
            
            preds = model(images)
            all_preds = torch.cat(
                (all_preds, preds)
                ,dim=0
            )
            true_preds = torch.cat(
                (true_preds, batch[1].to(device) )
                ,dim=0
            )
            del images
            del preds
            if get_activations:
                a1, a2, a3, a4 = model.get_activations()
                if first:
                    first = False
                    a1d, a2d, a3d, a4d = a1, a2, a3, a4
                else:
                    a1d = torch.cat((a1d,a1), 0)
                    a2d = torch.cat((a2d,a2), 0)
                    a3d = torch.cat((a3d,a3), 0)
                    a4d = torch.cat((a4d,a4), 0)
                activations = [a1d.to('cpu'), a2d.to('cpu'), a3d.to('cpu'), a4d.to('cpu')]
            torch.cuda.empty_cache()
    return all_preds, true_preds, activations

def get_performance_metrics(classes, true, preds):
    # TP TN FP FN
    dums = np.zeros((len(classes),4))
    acc = []
    rec = []
    spec = []
    prec = []
    mcc = []
    f1 = []
    for label in range(len(classes)):
        for tag in range(len(true)):
            if preds[tag].argmax() == label:
                if true[tag]== label:
                    dums[label,0]+=1
                else:
                    dums[label,2]+=1
            else:
                if true[tag]== label:
                    dums[label,3]+=1
                else:
                    dums[label,1]+=1
                    
        TP = dums[label,0]
        TN = dums[label,1]
        FP = dums[label,2]
        FN = dums[label,3]
        acc.append((TP+TN)/(TP+TN+FP+FN+0.000001))
        rec.append(TP/(TP+FN+0.000001))
        spec.append(TN/(TN+FP+0.000001))
        prec.append(TP/(TP+FP+0.000001))
        mcc.append(((TP*TN)-(FP*FN))/np.sqrt(float((TP+FN)*(TN+FP)*(TP+FP)*(TN+FN))+0.00001))
        f1.append(2*prec[label]*rec[label]/(prec[label]+rec[label]+0.000001))
    
    TP = sum(dums[:,0])
    TN = sum(dums[:,1])
    FP = sum(dums[:,2])
    FN = sum(dums[:,3])
    gacc = (TP+TN)/(TP+TN+FP+FN+0.000001)
    grec = TP/(TP+FN+0.000001)
    gspec = TN/(TN+FP+0.000001)
    gprec = TP/(TP+FP+0.000001)
    
    gmcc_d =np.sqrt(float((TP+FN)*(TN+FP)*(TP+FP)*(TN+FN))+0.000000001)
    gmcc = ((TP*TN)-(FP*FN)) / gmcc_d
    if np.isnan(gmcc):
        gmcc = -1
    
    gf1 = 2*gprec*grec/(gprec+grec)
    t = dums[:,0]+dums[:,3]
    p = dums[:,0]+dums[:,2]
    
    Mmcc_d = (np.sqrt(float((len(true)**2)-np.dot(p,p))) * np.sqrt(float((len(true)**2)-np.dot(t,t))))
    if Mmcc_d == 0 :
        Mmcc = 0
    else:
        Mmcc = (TP*len(true) - np.dot(t,p)) / Mmcc_d  
        
    if np.isnan(Mmcc):
        Mmcc = -1
    
    d = {'Value' : [TP/len(true), sum(acc)/len(acc), gacc, sum(rec)/len(rec), sum(spec)/len(spec), sum(prec)/len(prec), sum(f1)/len(f1), Mmcc],
          }
    metrics = pd.DataFrame(data = d, index = ['cat-acc', 'macro-acc', 'micro-acc', 'macro-rec', 'spec', 'macro-prec', 'micro-f1', 'mcc'])
    
    return metrics, dums

def get_CM(true, preds):
    
    size = preds.size()[1]
    cm = np.zeros((size,size))
    for i in range(len(preds)):
        cm[int(true[i].item()), preds[i].argmax().item()]+=1
        
    return cm
    
def Test_on_TestPartition(directory, DATABASE, device='cpu'):
    
    Exp_dir = directory + '/GenCNN/' + DATABASE
    
    directory = directory + '/' + DATABASE
    classes = Dataset_creator.get_classes(directory)
    min_shape = Dataset_creator.get_size(directory + '/' + 'test')
    shape = 224
    batch = 32
    if min(min_shape) > 224 :
        shape = 224 * (min(min_shape) // 224 )
        batch = int(32 / ((min(min_shape) // 224)**1))
    if min(min_shape) < 224 :
        shape = 224 / (224 // min(min_shape))
        batch = int(32 * (224 // (min(min_shape))**1))
        
    train_set, valid_set, test_set = Dataset_creator.get_sets(directory, train_size=(shape,shape), train_crop=(shape,shape), valid_size=(shape,shape),
                                 test_size=(shape,shape))
    
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, num_workers=1,
    #                                             shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch, num_workers=1,
    #                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, num_workers=1,
                                                  shuffle=True)
    
    Best_genes_dum = pd.DataFrame(pd.read_csv(Exp_dir+'/'+'Best_genes.csv')).to_numpy()[:,1:].tolist()
    Best_genes = []
    for i in range(len(Best_genes_dum)):
        Best_genes.append(Best_genes_dum[i][0])
    B_model = Hyperparameters.get_model(Best_genes[0], classes)
    B_model.load_state_dict(torch.load(Exp_dir+'/'+'B_model.pt'))
    B_model.to(device)
    
    test_preds, test_labels = get_all_preds(B_model, test_loader, device)
    
    B_metrics, B_dums = get_performance_metrics(classes, test_labels, test_preds)
    print('B test metrics: \n', B_metrics)
    B_metrics.to_csv(Exp_dir+'/'+'B_test_metrics.csv')
    B_CM = get_CM(test_labels, test_preds)
    np.set_printoptions(suppress=True)
    print('B confusion matrix: \n', B_CM)
    
    return B_metrics, B_CM
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
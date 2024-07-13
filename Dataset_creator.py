# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:45:56 2023

@author: Anonymous Author
"""

import os
import shutil
import random
import glob
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision


def get_classes(main_dir): 
    classes = []
    os.chdir(main_dir)
    for name in os.listdir(main_dir):
        if os.path.isdir(name) and name not in ['train','valid','test']:
            classes.append(name)
            
    return classes



def create_kinds_of_sets_folders(classes, main_dir, kinds_of_sets = ['train', 'valid', 'test'], 
                                 proportion =  [0.6, 0.2, 0.2]):
    os.chdir(main_dir)
    for kind in kinds_of_sets:
        os.chdir(main_dir)
        if os.path.isdir(kind) is False:
            os.makedirs(kind)
            for label in classes:
                samples = len(os.listdir(os.chdir(os.path.join(main_dir,label))))
                os.chdir(os.path.join(main_dir,kind))
                if os.path.isdir(label) is False:
                    os.makedirs(label)
                    
                    if kind == 'train':
                        os.chdir(os.path.join(main_dir,label))
                        for c in random.sample(glob.glob('*.*'), round(samples*proportion[0])):
                            shutil.move(c, os.path.join(main_dir,kind,label))
                            
                    elif kind == 'valid':
                        os.chdir(os.path.join(main_dir,label))
                        for c in random.sample(glob.glob('*.*'), round(samples*proportion[1]/(proportion[1]+proportion[2]))):
                            shutil.move(c, os.path.join(main_dir,kind,label))
                            
                    else:
                        os.chdir(os.path.join(main_dir,label))
                        for c in random.sample(glob.glob('*.*'), samples):
                            shutil.move(c, os.path.join(main_dir,kind,label))
                            
                            
def make_csv_file(data_dir, labels):
    os.chdir(data_dir)
    images_and_labels = []
    for label in labels:
        dum_dir = os.path.join(data_dir, label)
        files = os.listdir(dum_dir)
        for file in files:
            if '.ini' not in file:
                images_and_labels.append([file,label])
    csv_file = pd.DataFrame(images_and_labels)
    csv_file.to_csv('images_and_labels.csv')

class DummiDataset(Dataset):
    def __init__(self, subset, transform = None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)
    
class MedImageDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.files = pd.read_csv(os.path.join(data_dir, csv_file))
        self.transform = transform
        self.data_dir = data_dir
        self.labels = pd.Series.unique(self.files.iloc[:, 2])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.files.iloc[index, 2], self.files.iloc[index, 1])
        image = io.imread(img_path)
        y_label = torch.tensor(int(np.where(self.labels==self.files.iloc[index, 2])[0]))
        if image.ndim != 3:
            image = np.dstack((image,image,image))
        if self.transform:
            image = self.transform(image)
        return (image, y_label)

def get_size(images_directory):
    
    files = pd.read_csv(os.path.join(images_directory, 'images_and_labels.csv'))
    shapes = []
    for index in range(len(files)):
        img_path = os.path.join(images_directory, files.iloc[index, 2], files.iloc[index, 1])
        image = io.imread(img_path)
        shapes.append([image.shape[0],image.shape[1]])
        
    return min(shapes)
        

def get_sets(directory, train_size=(224,224), train_crop=(188,188), valid_size=(224,224),
                             test_size=(360,360)):
    
    main_dir = directory
    
    classes = get_classes(main_dir)
    create_kinds_of_sets_folders(classes, main_dir)
    
    train_dir = main_dir + '/train'
    valid_dir = main_dir + '/valid'
    test_dir = main_dir + '/test'
    
    make_csv_file(train_dir, classes)
    make_csv_file(valid_dir, classes)
    make_csv_file(test_dir, classes)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_set = MedImageDataset(csv_file = 'images_and_labels.csv', 
                  data_dir = train_dir, 
                  transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(train_size),
                transforms.RandomCrop(size=train_crop),
                transforms.RandomRotation(degrees=(90)),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.ColorJitter(brightness=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            )
    
    valid_set = MedImageDataset(csv_file = 'images_and_labels.csv', 
                  data_dir = valid_dir, 
                  transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(valid_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])
            )
    
    test_set = MedImageDataset(csv_file = 'images_and_labels.csv', 
                  data_dir = test_dir, 
                  transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(test_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])
            )

    return train_set, valid_set, test_set
    
    
    
    
    
    
    

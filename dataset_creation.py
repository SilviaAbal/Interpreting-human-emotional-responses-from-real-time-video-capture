#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:20:14 2022

@author: sabal
"""

import torch
import cv2
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import glob

    
"""
This class allows us to be able to manage the database.

This is done because it returns, apart from the image and its corresponding label, 
two particular variables: img_path and person. 
    - img_path: indicates he path of the image. This allows to save the image depending on whether 
        it has been classified well or badly.
    - person: indicates the name of the person identified in the image, if identified. For this,
        it is important that the name of the images follow the same syntax.

"""

class CustomImageDataset(Dataset):
    def __init__(self, img_path, type_dataset, transform=None, target_transform=None):
        
        self.transform = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.target_transform = target_transform
        self.imgs_path = img_path
        file_list = glob.glob(self.imgs_path + "*")

        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])

        if type_dataset=="Fer2013":
            self.class_map = {"Angry":0, "Disgusted":1, "Fearful":2, "Happy":3, "Sad":4, "Surprised":5, "Neutral":6}
        elif type_dataset=="own":
            self.class_map = {"neutral":0, "positive":1, "negative":2}
        elif type_dataset=="AffectNet":
            self.class_map = {"neutral":0, "happy":1, "sad":2, "surprise":3, "fear":4, "anger":5 , "disgust":6, "contempt":7}
            
        self.img_dim = (224, 224) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)

        if (img.shape[2]==1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        class_id = self.class_map[class_name]
        
        image_tensor = self.transform(img) 
        image_norm = self.normalize(image_tensor)
        
        label = torch.tensor(class_id)

        person = (img_path.split("/")[-1]).split("_")[0]
        if self.target_transform:
            label = self.target_transform(label)
        

        return image_norm, label, img_path, person
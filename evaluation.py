#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:48:54 2022

@author: sabal

Script with auxiliary functions for calculating network 
performance and displaying results, among other things.

"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools
    
# Emotion dictionaries
Own_emotion_dict = {0: "neutral", 1: "positive", 2: "negative"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def computeAUROC (scores, labels):
    """
    Compute Area Under the Curve (AUC).
    
    """
    
    aucs = np.zeros((3,))

    scores_positive = scores[:,1]
    labels_positive = (labels == 1).astype(int) 
    aucs[1]=metrics.roc_auc_score(labels_positive, scores_positive)

    scores_negative = scores[:,2]
    labels_negative = (labels == 2).astype(int) 
    aucs[2]=metrics.roc_auc_score(labels_negative, scores_negative)
    
    scores_neutral = scores[:,0]
    labels_neutral = (labels == 0).astype(int) 
    aucs[0]=metrics.roc_auc_score(labels_neutral, scores_neutral)
               
    return aucs


def errors(scores, labels, images, path_img, path_errores, path_success, save): 
    """
    Images that have been misclassified are saved as errors in the folder 'errors', 
    as well as there is the option to save well classified images in the folder 'success'.
    
    """
        
    # Turn off gradients for validation, saves memory and computations
    top_p, top_class = scores.topk(1, dim=1)

    # Store wrongly predicted images
    wrong_idx = (top_class != labels.view(top_class.shape[0], 1)).nonzero()[:, 0]
    wrong_samples = images[wrong_idx]
    wrong_preds = top_class[wrong_idx]
    actual_preds = labels.view_as(top_class)[wrong_idx]
    
    right_idx =  (top_class == labels.view(top_class.shape[0], 1)).nonzero()[:, 0]
    right_samples = images[right_idx]
    right_preds = top_class[right_idx]
    
    if save == 'errors' or save =='all':
        
        for i, val  in enumerate(wrong_idx):
            sample = wrong_samples[i]
            wrong_pred = wrong_preds[i]
            actual_pred = actual_preds[i]
    
            sample = sample * 0.3081
            sample = sample + 0.1307
            sample = sample * 255.
            
            if 'train' not in path_img or 'val' not in path_img:    
    
                imagen = cv2.imread(path_img[val], cv2.IMREAD_GRAYSCALE)
            else:
    
                imagen = np.transpose(sample.cpu().numpy(), (1,2,0))
                print(imagen.shape)
            
            idx_img = (path_img[val].split(".jpg")[0]).split("/")[-1]
            
            name_img = path_errores + idx_img + '-wrong_pred_{}_actual_{}.jpg'.format(Own_emotion_dict[wrong_pred.item()], Own_emotion_dict[actual_pred.item()])
            cv2.imwrite(name_img, imagen)
    if save == 'success' or save == 'all':
         for i, val  in enumerate(right_idx):
            sample = right_samples[i]
            right_pred = right_preds[i]

            sample = sample * 0.3081
            sample = sample + 0.1307
            sample = sample * 255.
            
            if 'train' not in path_img or 'val' not in path_img:    
    
                imagen = cv2.imread(path_img[val], cv2.IMREAD_GRAYSCALE)
            else:
    
                imagen = np.transpose(sample.cpu().numpy(), (1,2,0))
                print(imagen.shape)
            
            idx_img = (path_img[val].split(".jpg")[0]).split("/")[-1]
            
            name_img = path_success + idx_img + '-right_pred_{}.jpg'.format(Own_emotion_dict[right_pred.item()])
            cv2.imwrite(name_img, imagen)

    return 

def evaluation(scores, labels): 
    """
    Funtion that returns the accuracy of the model.

    Returns
    -------
    accuracy : accuracy of the model.
    class_acc : accuracy of each of the categories of the model.
    cmt : confusion matrix.

    """

    accuracy = 0
    cmt=torch.zeros(3,3, dtype=torch.int64)
    top_p, top_class = scores.topk(1, dim=1)
    equals = (top_class == labels.view(top_class.shape[0], 1))
    accuracy += equals.sum()/len(equals)
    
    stacked = torch.stack((labels.view(top_class.shape[0], 1),top_class),dim=1)
    
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
        
    if cmt[0].sum() == 0: 
        neutral_acc =(cmt[0][0]).item()
    else: 
        neutral_acc =(cmt[0][0]/(cmt[0].sum())).item()
    
    if cmt[1].sum() == 0: 
        positive_acc =(cmt[1][1]).item()
    else: 
        positive_acc =(cmt[1][1]/(cmt[1].sum())).item()
    
    if cmt[2].sum() == 0:
        negative_acc =(cmt[2][2]).item()
    else:
        negative_acc =(cmt[2][2]/(cmt[2].sum())).item()
    
    class_acc = [neutral_acc, positive_acc,negative_acc]
    
    return accuracy,class_acc,cmt

def performance(model, test_data, save_errors, numSamples, save, path_errors, path_success): 
    """
    Function that returns the performance of the specified model
    
    Parameters
    ----------
    model : Model epoch.
    test_data : Data loader.
    save_errors : (Boolean). Indicates whether errors are to be saved.
    numSamples : (Int). Nuber of samples.
    save : (String). Posibilities:
        -'errors': only misclassified images are saved
        -'success': only well classified images are saved
        -'all': all images are saved in their corresponding folder
    
    Returns
    -------
    acc_emotion_dict : accuracy of each category
    cmt : confusion matrix
    aucs : area under the curve
    
    """
    accuracy = 0
    contSamples = 0
    cmt=torch.zeros(3,3, dtype=torch.int64)
    outputs_m=np.zeros((numSamples,3),dtype=float)
    labels_m=np.zeros((numSamples,),dtype=int)   
    
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
    
        for images, labels, path_img, _ in test_data:
            images, labels = images.to(device), labels.to(device)  
            path_img = list(path_img)
            batchSize = labels.shape[0]
            
            logprobs = model.forward(images)
            top_p, top_class = logprobs.topk(1, dim=1)
            equals = (top_class == labels.view(images.shape[0], 1))
            accuracy += equals.sum()
    
            stacked = torch.stack((labels.view(images.shape[0], 1),top_class),dim=1)
            
            for p in stacked:
                tl, pl = p.tolist()
                cmt[tl, pl] = cmt[tl, pl] + 1
                        
            if save_errors == True: 
                errors(logprobs, labels, images, path_img, path_errors, path_success, save)
     
            outputs_m [contSamples:contSamples+batchSize,...]=logprobs.cpu().numpy()
            labels_m [contSamples:contSamples+batchSize,...]=labels.cpu().numpy()
            
            contSamples+=batchSize
    
        aucs=computeAUROC(outputs_m,labels_m)
    
    neutral_acc =(cmt[0][0]/(cmt[0].sum())).item()
    positive_acc =(cmt[1][1]/(cmt[1].sum())).item()
    negative_acc =(cmt[2][2]/(cmt[2].sum())).item()
    acc_emotion_dict ={"neutral acc":neutral_acc, "positive acc":positive_acc, "negative acc":negative_acc}
    
    return accuracy/numSamples,acc_emotion_dict, cmt, aucs
        
def plotConfusionMatrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Function that creates the graphical representation of the confusion matrix.

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plotAcc(val_acc, train_acc, person):
    """
    Creates the graph that represents the accuracy vs. epoch distinguishing between people
    
    """
    # Acc Plot
    plt.plot(train_acc,label='Training Acc '+person)
    plt.plot(val_acc, marker='o',label='Validation Acc '+person)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title('Training and validation acc vs. epochs')
      
def findMaxList(list):
    """
    Returns the maximum length found in a list of lists.
    
    """
    list_len = [len(i) for i in list]
    return max(list_len)

def extract(lst,idx,category):
    """
    Returns a list as the list of idx elements within a sublist.
    
    """
    if category: 
        ret = [item[idx] for item in lst]
    else: 
        ret = [item[idx].item() for item in lst]
        
    return ret 
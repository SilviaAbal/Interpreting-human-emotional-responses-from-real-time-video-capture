# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:31:06 2022

@author: Silvia

Script in which functions are defined to relabel the images of the AffectNet 
database following our criteria of positive, negative and neutral categories.

"""

import os
import numpy as np
from PIL import Image
import pdb

root_path = os.getcwd()
Own_dict = {"neutral":0, "positive":1, "negative":2}
AffectNet_dict = {"neutral":0, "happy":1, "sad":2, "surprise":3, "fear":4, "anger":5 , "disgust":6, "contempt":7}
AffectNet_relabelled = {0: "neutral", 1: "positive", 2: "negative",3:"no",4:"negative",5:"negative",6:"negative",7:"negative",8:"no",9:"no",10:"no"}


def createFolds (path):
    """
    Funtion that create the required folds to correctly save the AffectNet images

    Parameters
    ----------
    path : folder path to be created

    Returns
    -------
    None.

    """
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)
    return

def saveImages (dire, save_path, mode):
    """
    Function that saves the images in the folders of the categories positive,
    negative and neutral following the relabelling criteria defined 
    in the variable 'AffectNet_relabelled'. the images in the folders train 
    and val will be distinguished in the image names.

    Parameters
    ----------
    dire : path where the images come from 
    save_path : path where images are stored 
    mode : train or val
    Returns
    -------
    None.

    """
    for root, dirs, files in os.walk(dire+'/images/'):
        print(files)
        for file in files:
            pdb.set_trace()
            number_file = file.split(".")[0]
            npy_file_emo = dire+ "/annotations/"+ number_file+"_exp.npy"
            emotion = np.load(npy_file_emo)
            
            if AffectNet_relabelled[int(emotion)] != "no":
                img = Image.open(root+file)
                im_save = save_path+"/"+AffectNet_relabelled[int(emotion)]+'/'+str(number_file)+'_'+mode+'.jpg'
                img.save(im_save)

save_path = root_path + '/AffecNet_relabelled'
save_path_positive = root_path + '/AffecNet_relabelled/positive'
save_path_negative = root_path + '/AffecNet_relabelled/negative'
save_path_neutral = root_path + '/AffecNet_relabelled/neutral'

createFolds(save_path)
createFolds(save_path_positive)
createFolds(save_path_negative)
createFolds(save_path_neutral)

dire_train = root_path + '/train_set'
dire_val = root_path + '/val_set'

saveImages(dire_train, save_path, 'train')
saveImages(dire_val, save_path, 'val')
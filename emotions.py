#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:08:34 2022

@author: sabal
"""

import numpy as np
import argparse
import cv2
import torch
import os
import config as cfg
import matplotlib.pyplot as plt
import pandas as pd 
import torch.nn.functional as F
from torch.utils import data
from datetime import date
import datetime
from dataset_creation import CustomImageDataset
from evaluation import computeAUROC,evaluation,performance,plotConfusionMatrix,plotAcc,findMaxList,extract
import time
import re
from weak_loss_layer.weak_loss import WeakLoss



root_path = os.getcwd()

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display. train: finetunning the model / display: detects emotions in real time via webcam")
ap.add_argument('--num_epoch', type=int, default=cfg.NUM_EPOCH, help="(int) Number of epochs for training the network. Default: 15.")
ap.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE, help="(int) Batch size for the training of the network. Default: 64.")
ap.add_argument('--lr', type=float, default=cfg.LR, help="(float) Learning rate. Default: 6e-5.")
ap.add_argument('--gamma', type=float, default=cfg.GAMMA, help="(float) Discount rate of future rewards. Default: 0.")
ap.add_argument('--percentage', type=float, default=cfg.PERCEN, help="(float) Percentage of AffectNet images used for finetunning the model [0,100]. Default: 0.05 (=0.0005%)")
ap.add_argument('--weight_decay', type=float, default=cfg.WEIGHT_DECAY, help="(float) L2 regularization method. Default: 1")
ap.add_argument('--results_per_person', type=bool, default=cfg.RESULTS_PER_PERSON, help="(boolean) Display the results obtained per person identified in the dataset. Default: False")
ap.add_argument('--pretrained_model_display', type=int, default=cfg.MODEL_DISPLAY, help="(int) Selection of pretrained model used (1,2,3). Default: 1")

args = ap.parse_args()

mode = args.mode
results_person = args.results_per_person

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

eye_cascade = cv2.CascadeClassifier(root_path + '/classifier/haarcascade_eye.xml')
facecasc = cv2.CascadeClassifier(root_path + '/classifier/haarcascade_frontalface_default.xml')
 
# Emotion dictionaries
Own_emotion_dict = {0: "neutral", 1: "positive", 2: "negative"}

# Train the network 
if mode == "train": 

    folds = ['1fold/','2fold/','3fold/']
    
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    percen = args.percentage

    dataset_path = root_path + '/AffectNet_relabelled/'
    dataset_path_fold = root_path + '/nfold/3-fold/'
    
    for fold in folds:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        torch.manual_seed(0)
        
        # Train and validation dataloaders creation
        dataset_path_fold_ = dataset_path_fold + fold
        dataset_train = CustomImageDataset(img_path=dataset_path_fold_+'train/', type_dataset='own')
        dataset_val = CustomImageDataset(img_path=dataset_path_fold_+'test/', type_dataset='own')

        if args.percentage != 0: 
            dataset = CustomImageDataset(img_path=dataset_path, type_dataset='own')
            dataset_train_orig,_ = torch.utils.data.random_split(dataset, [int(round(len(dataset)*(percen/100))), int(round(len(dataset)*((100-percen)/100)))],generator=torch.Generator().manual_seed(0))
            dataset_train = dataset_train + dataset_train_orig
            
        train_loader = data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True,num_workers=2,generator=torch.Generator().manual_seed(0))
        valid_loader = data.DataLoader(dataset_val, batch_size = batch_size, shuffle = False,generator=torch.Generator().manual_seed(0))
        
        name_folder = dataset_path_fold_.split("/")[-2]
        name_split = dataset_path_fold_.split("/")[-3]
        print(name_folder)
        
        dataloaders = {'train' : train_loader, 'val': valid_loader}
        dataset_sizes = {'train': len(dataset_train), 'val': len(dataset_val)}
        
        current_time = datetime.datetime.now()

        # Folders creation to save the results; Models epoch, errors (images missclassified), successes (images well classiffied) are saved.
        path_results = root_path + "/results/" +"test_finetunning_"+str(percen)+"%_old_dataset"+name_split+"_"+name_folder+"_"+str(current_time.day)+"-"+str(current_time.month)+"__"+str(current_time.hour)+"h_"+str(current_time.minute)+"min"
        
        try:
            os.mkdir(path_results)
        except OSError:
            print ("Creation of the directory %s failed" % path_results)
        else:
            print ("Successfully created the directory %s " % path_results)
            
        path_model_epoch =path_results+'/models_epoch'
        try:
            os.mkdir(path_model_epoch)
        except OSError:
            print ("Creation of the directory %s failed" % path_model_epoch)
        else:
            print ("Successfully created the directory %s " % path_model_epoch)
        
        path_errors =path_results+'/errors/'
        try:
            os.mkdir(path_errors)
        except OSError:
            print ("Creation of the directory %s failed" % path_errors)
        else:
            print ("Successfully created the directory %s " % path_errors)
          
        path_success =path_results+'/success/'
        try:
            os.mkdir(path_success)
        except OSError:
            print ("Creation of the directory %s failed" % path_success)
        else:
            print ("Successfully created the directory %s " % path_success)
                  
        def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epoch, nlabels=3):
            """
            Train and validation process of the network.
            
            """
            
            since = time.time()
        
            train_loss = []
            val_loss = []
        
            train_auc = []
            val_auc = []
            
            train_auc_class = []
            val_auc_class = []
            
            cmt_train = []
            cmt_val = []
            cmt = []
                
            train_acc = []
            val_acc = []
            train_acc_class = []
            val_acc_class= []
            
            val_acc_person = []
            val_acc_class_person = []
            cmt_val_person = []
            train_acc_person = []
            train_acc_class_person = []
            cmt_train_person = []
            names_person = []
            
            txt_name = path_results + '/train_&_valid_loss.txt'
            
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
        
                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode
        
                    numSamples = dataset_sizes[phase]
                    outputs_m=np.zeros((numSamples,nlabels),dtype=float)
                    labels_m = np.zeros((numSamples,),dtype=int)
                    
                    persons_m = []
                    running_loss = 0.0
                  
                    contSamples=0

                    # Iterate over data.                    
                    for inputs, labels, path_img, persons in dataloaders[phase]:
                            
                        inputs, labels = inputs.to(device).float() , labels.to(device)  
                        path_img = list(path_img)
                        persons = list(persons)
                   
                        #Batch Size
                        batchSize = labels.shape[0]
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        
                        weights=torch.from_numpy(np.ones((batchSize,1),dtype=int))
                        weights=weights.to(device)
        
                        labels_ = torch.reshape(labels, (batchSize,1)) 
                        
                        loss_pos=0
                        loss_neu=0
                        loss_neg=0
            
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            outs_positive = outputs[labels==0]
                            outs_neutral = outputs[labels==1]
                            outs_negative = outputs[labels==2]
                                           
                            if 0 in labels: 
                                loss_pos = criterion(outs_positive, labels_[labels==0],weights[labels==0])
    
                            if 1 in labels:
                                loss_neu = criterion(outs_neutral, labels_[labels==1],weights[labels==1])
    
                            if 2 in labels:
                                loss_neg = criterion(outs_negative, labels_[labels==2],weights[labels==2])
    
                            loss = loss_pos + loss_neu + loss_neg
        
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                                torch.cuda.empty_cache()
                        
                        #Apply a softmax to the output
                        outputs = F.softmax(outputs.data,dim=1)
                        running_loss += loss.item()
        
                        # Store outputs and labels 
                        outputs_m [contSamples:contSamples+batchSize,...]=outputs.cpu().numpy()
                        labels_m [contSamples:contSamples+batchSize,...]=labels.cpu().numpy()
                        if results_person == True:
                            persons_m.append(persons)
                        
                        contSamples+=batchSize
                    if phase == 'train':
                        scheduler.step()
    
                    epoch_loss = running_loss / dataset_sizes[phase]
        
                    aucs=computeAUROC(outputs_m,labels_m)
                    
                    outs = torch.tensor(outputs_m)
                    labs = torch.tensor(labels_m)
                    
                    if results_person == True:
                        flat_list = [item for sublist in persons_m for item in sublist]
                        
                        for idx in persons: 
                            if not re.search('\d+', idx):
                                if not names_person: 
                                    names_person.append(idx)
         
                                else:
                                    if idx not in names_person:
                                        names_person.append(idx)
                                print(idx)
        
        
                        acc_person = list(torch.zeros((len(names_person))))
                        acc_class_person = list(torch.zeros((len(names_person))))
                        cmt_person = []
                        for i in names_person: 
                            cmt_person.append(torch.zeros(3,3, dtype=torch.int64))
        
                        for idxn, nom in enumerate(names_person): 
                            
                            indices = [i for i, x in enumerate(flat_list) if x == nom]
                            outs_ = np.zeros((len(indices),nlabels),dtype=float)
                            labs_ = np.zeros((len(indices),),dtype=int)
                            
                            for i,v in enumerate(outs): 
                                for idx, val in enumerate(indices): 
                                    if i==val: 
                                       
                                        outs_[idx] = outs[i]
                                        labs_[idx]= labs[i]
                            
                            outs_ = torch.tensor(outs_)
                            labs_ = torch.tensor(labs_)
                        
                            acc_person[idxn], acc_class_person[idxn], cmt_person[idxn] = evaluation(outs_, labs_)
                                
                    epoch_acc_global, epoch_acc_class_global, cmt = evaluation(outs, labs)
    
                    #And the Average AUC
                    epoch_auc = aucs.mean()
                                 
                    PATH = path_model_epoch+'/model_epoch'+ str(epoch)+'.pt'
                    torch.save(model, PATH)
                    
                    print('\n{} Loss: {:.4f}\n'.format(phase, epoch_loss))
                    print('{} AUC neutral: {:.4f} positive: {:.4f} negative: {:.4f} mean: {:.4f}'.format(
                        phase, aucs[0], aucs[1], aucs[2], epoch_auc))
                    print('{} acc: {:.4f}, acc per class: {}\n'.format(phase,epoch_acc_global,epoch_acc_class_global))
                    
                    if results_person == True:
                        for idxn, nom in enumerate(names_person):
                            print('{} {} acc: {:.4f}, acc per class: {}'.format(nom,phase,acc_person[idxn],acc_class_person[idxn]))
            
                    with open(txt_name, 'a') as f:
                        if phase == 'train':    
                            f.write("\n--------- Epoch "+ str(epoch)+ "--------- ")
                        f.write("\n {} loss: {}".format(phase, epoch_loss))
                        
                    if phase == 'train': 
                      train_loss.append(epoch_loss)
                      train_auc.append(epoch_auc)
                      train_auc_class.append(aucs) 
                      train_acc.append(epoch_acc_global)
                      train_acc_class.append(epoch_acc_class_global)
                      cmt_train.append(cmt)
                      if results_person == True:
                          train_acc_person.append(acc_person)
                          train_acc_class_person.append(acc_class_person)
                          cmt_train_person.append(cmt_person)
                      
                    if phase == 'val': 
                      val_loss.append(epoch_loss)
                      val_auc.append(epoch_auc)
                      val_auc_class.append(aucs)
                      val_acc.append(epoch_acc_global)
                      val_acc_class.append(epoch_acc_class_global)
                      cmt_val.append(cmt)
                      if results_person == True:
                          val_acc_person.append(acc_person)
                          val_acc_class_person.append(acc_class_person)
                          cmt_val_person.append(cmt_person)
                      
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            if results_person == True:
                return model,train_loss,val_loss,train_acc, val_acc,train_acc_class,val_acc_class, train_auc, val_auc, train_auc_class, val_auc_class, cmt_train, cmt_val, train_acc_person,train_acc_class_person,cmt_train_person, val_acc_person,val_acc_class_person,cmt_val_person,names_person
            else: 
                return model,train_loss,val_loss,train_acc, val_acc,train_acc_class,val_acc_class, train_auc, val_auc, train_auc_class, val_auc_class, cmt_train, cmt_val
        
        
            
            
        # FINETUNNING
        model_ft = torch.load(root_path + '/pretrained_models/model_finetunning.pt')
        model_ft = model_ft.to(device)
        
        LOSS_BOUNDS = cfg.LOSS_BOUNDS
        bounds = torch.tensor(LOSS_BOUNDS,dtype=torch.float32)
        fg_slack = cfg.LOSS_SLACK 
        weakloss = WeakLoss(bounds,fg_slack=fg_slack)
    
        # Observe that all parameters are being optimized
        lr = args.lr
        weight_decay = args.weight_decay
        gamma=args.gamma
        
        optimizer_ft = torch.optim.Adam(model_ft.parameters(),lr,weight_decay = weight_decay)
        exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma = gamma)
        
        # TRAIN THE MODEL 
        if results_person == True:
            model_ft, train_loss, valid_loss,train_acc, val_acc,train_acc_class,val_acc_class, train_auc, val_auc, train_auc_class, val_auc_class, cmt_train, cmt_val,train_acc_person,train_acc_class_person,cmt_train_person, val_acc_person,val_acc_class_person,cmt_val_person,names_person = train_model(model_ft, weakloss, optimizer_ft, exp_lr_scheduler, num_epochs=num_epoch, nlabels=len(Own_emotion_dict))
        else: 
            model_ft, train_loss, valid_loss,train_acc, val_acc,train_acc_class,val_acc_class, train_auc, val_auc, train_auc_class, val_auc_class, cmt_train, cmt_val = train_model(model_ft, weakloss, optimizer_ft, exp_lr_scheduler, num_epochs=num_epoch, nlabels=len(Own_emotion_dict))
        
        today = date.today()
       
        # Save results 
        model_name = path_results + "/model_reward_"+str(num_epoch)+today.strftime("_%d_%m_%Y")+".h5"
        torch.save(model_ft.state_dict(), model_name)
        
        txt_name = path_results + '/neural_network_used.txt'
        with open(txt_name, 'a') as f:
            f.write("\nlr: "+str(lr))
            f.write("\nweight_decay: "+str(weight_decay))
            f.write("\ngamma: "+str(gamma))
                
        # Train accuracy
        train_accuracy, train_accuracy_class, cmt_accuracy_train, aucs_train = performance(model_ft,train_loader, save_errors = False, numSamples = len(dataset_train), save = False, path_errors=path_errors, path_success=path_success)
        print("\n\033[1mTrain Accuracy:\033[0m %f" %(train_accuracy))
        print("\n\033[1mTrain Accuracy per class:\n%s" %(train_accuracy_class))
        print('Train AUC neutral: {:.4f} positive: {:.4f} negative: {:.4f} mean: {:.4f}\n'.format(aucs_train[0], aucs_train[1], aucs_train[2], aucs_train.mean()))
        
        # Validation accuracy
        valid_accuracy, valid_accuracy_class, cmt_accuracy_valid, aucs_valid = performance(model_ft,valid_loader, save_errors = True, numSamples = len(dataset_val), save = 'all', path_errors=path_errors, path_success=path_success) 
        print("\nValid Accuracy:\033[0m %f" %(valid_accuracy))
        print("\033[1mValid Accuracy per class :\n%s"  %(valid_accuracy_class))
        print('Valid AUC neutral: {:.4f} positive: {:.4f} negative: {:.4f} mean: {:.4f}\n'.format(aucs_valid[0], aucs_valid[1], aucs_valid[2], aucs_valid.mean()))
        
        # Loss Plot
        optim_epoch_val = min(valid_loss)
        optim_epoch = valid_loss.index(optim_epoch_val)
        
        plt.figure(figsize=(10,7))
        plt.plot(train_loss,label='Training Loss')
        plt.plot(valid_loss, marker='o',label='Validation Loss')
        plt.scatter(optim_epoch,
                    optim_epoch_val, c='r', marker='o', s=200, label='Optimal epoch')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and validation loss vs. epochs')
        
        
        fig_name = path_results + "/train_validation_loss_vs_"+str(num_epoch)+today.strftime("_%d_%m_%Y")+".jpg"
        plt.savefig(fig_name)
        
        
        # Acc Plot
        optim_epoch_val = max(val_acc)
        optim_epoch = val_acc.index(optim_epoch_val)
        
        plt.figure(figsize=(10,7))
        plt.plot(train_acc,label='Training Acc')
        plt.plot(val_acc, marker='o',label='Validation Acc')
        plt.scatter(optim_epoch,
                    optim_epoch_val, c='r', marker='o', s=200, label='Optimal epoch')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.title('Training and validation acc vs. epochs')
        
        
        fig_name = path_results + "/train_validation_acc_vs_"+str(num_epoch)+today.strftime("_%d_%m_%Y")+".jpg"
        plt.savefig(fig_name)
        
        
        d = {'train accuracy':[float(train_accuracy)], 'valid accuracy':[float(valid_accuracy)]}
        df = pd.DataFrame(data=d)
        dict_results = path_results + "/accuracy_results.csv"
        df.to_csv(dict_results, index=False) 
        
        db = {'train accuracy':[str(train_accuracy_class)], 'valid accuracy':[str(valid_accuracy_class)]}
        df_b = pd.DataFrame(data=db)
        dict_results_per_class = path_results + "/accuracy_per_class_results.csv"
        df_b.to_csv(dict_results_per_class, index=False) 
        
        d = {'train AUC':[float(aucs_train.mean()),str(aucs_train)], 'valid AUC':[float(aucs_valid.mean()), str(aucs_valid)]}
        df = pd.DataFrame(data=d)
        dict_results = path_results + "/AUC_results.csv"
        df.to_csv(dict_results, index=False) 
        
        classes = list(Own_emotion_dict.values())
        
        plt.figure(figsize=(10,10))
        plotConfusionMatrix(cmt_accuracy_train,  classes)
        fig_name = path_results + "/cmt_train.jpg"
        plt.savefig(fig_name)
        
        plt.figure(figsize=(10,10))
        plotConfusionMatrix(cmt_accuracy_valid,  classes)
        fig_name = path_results + "/cmt_val.jpg"
        plt.savefig(fig_name)
    
        if results_person == True:
            max_list_val=findMaxList(val_acc_person)
            max_list_train=findMaxList(train_acc_person)
             
            for idx, val in enumerate(val_acc_person): 
                while len(val)<max_list_val:
                    val_acc_person[idx].append(np.array(float('nan')))
                    val_acc_class_person[idx].append([0,0,0])
            for idx, val in enumerate(train_acc_person): 
                while len(val)<max_list_train:
                    train_acc_person[idx].append(np.array(float('nan')))
                    train_acc_class_person[idx].append([0,0,0])
                 
            
            for i in range(0,max_list_train):
                val_acc = extract(val_acc_person,i, False)
                train_acc = extract(train_acc_person,i, False)
                if i == 0:
                    plt.figure(figsize=(10,7))
                plotAcc(val_acc, train_acc, names_person[i])
                        
                if i==max_list_train-1:
                    fig_name = path_results + "/train_validation_acc_vs_epoch_person"+today.strftime("_%d_%m_%Y")+".jpg"
                    plt.savefig(fig_name)
                    plt.show()
                
                train_acc_class = extract(train_acc_class_person,i, True)
                val_acc_class = extract(val_acc_class_person,i, True)
             
                train_name = "train acc "+names_person[i]
                val_name = "val acc "+names_person[i]
                
                train_name_class = "train acc per class "+names_person[i]
                val_name_class = "val acc per class "+names_person[i]
                
                data_df = {train_name:train_acc,val_name:val_acc}
                data_class_df = {train_name_class:train_acc_class,val_name_class:val_acc_class}
                if i == 0: 
                    df1 = pd.DataFrame(data=data_df)
                    df1_class = pd.DataFrame(data=data_class_df)
                else: 
                    df2 = pd.DataFrame(data=data_df)
                    df1 = pd.concat([df1,df2],axis=1)
                    
                    df2_class = pd.DataFrame(data=data_class_df)
                    df1_class = pd.concat([df1_class,df2_class],axis=1)
                
            dict_results = path_results + "/accuracy_results_per_person.csv"
            df1_class.to_csv(dict_results, index=False) 
            
            dict_results = path_results + "/accuracy_results_per_category_per_person.csv"
            df1_class.to_csv(dict_results, index=False) 
            

    

    
# Emotions will be displayed on your face from the webcam feed
if mode == "display":

    path_model = root_path + '/pretrained_models/model_display_'+str(args.pretrained_model_display)+'.pt'
    if torch.cuda.is_available():
        model = torch.load(path_model,map_location=torch.device('cuda:0'))
        model.to(device)
    else: 
        model = torch.load(path_model,map_location=torch.device('cpu'))
 
    model.eval()

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # start the webcam feed
    wait_frame = 0
    acum = 0
    count_frame = 0
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        wait_frame += 1
        
        if wait_frame>20:
            
            if not ret:
                break
           
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray) # detecto tb los ojos porque aveces no funciona bien la cara,
                                
                if len(eyes) > 0: 
                    count_frame += 1
                    img = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
                    
                    cropped_img = img.reshape((1, 3, w,w))   
                    cropped_img= torch.from_numpy(cropped_img)
                    cropped_img = cropped_img/255
                    cropped_img = cropped_img.to(device)
                    output = model(cropped_img)
                    prediction = int(torch.max(output.data, 1)[1].cpu().numpy())
                    
                    # print(prediction)
                    acum += prediction
                    if (count_frame==80):
                        prediction = round(acum/count_frame)
                        cv2.putText(frame, Own_emotion_dict[prediction], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        print(Own_emotion_dict[prediction])
                        break
                
            cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    

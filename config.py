#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:00:50 2022

@author: sabal

This file contains the configuration of the default parameters.

"""

BATCH_SIZE = 64
NUM_EPOCH = 15
PERCEN = 0.05 # Percentage of data from the AffectNet database that is used for finetunning
LR = 6e-5 # Learning rate
WEIGHT_DECAY = 1 
GAMMA = 0
RESULTS_PER_PERSON = False
MODEL_DISPLAY = 1
DISPLAY_MODE = "webcam"
FRAME_RATE_WEBCAM = 50
FRAME_RATE_VID = 3

# WEAKLOSS PARAMETERS 
LOSS_BOUNDS = [[0., 0.15], [0.85, 10.0]] # We require at least 85% of the frames to be correct.
LOSS_SLACK = 100.0 
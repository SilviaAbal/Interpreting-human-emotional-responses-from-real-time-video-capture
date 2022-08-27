# Interpreting human emotional responses from real-time video capture
The developed interface is part of the COMPANION-CM project [1], which aims to create more social assistive robots. One of the keys to achieving this goal is for these machines to be able to interpret and act in response to human responses generated by the robot's own actions. 

Convolutional Neural Networks (CNN) are used to carry out this part of the robot's AI, in order to extract features from video recordings. As for the training of the CNN, it is carried out with relevant training data, such as the AffectNet database [8], achieving a robust feature extraction system. AffectNet contains about 1 million facial images collected from the Internet, making it one of the largest databases of facial expressions in existence. In it, up to eleven categorical labels of emotions and non-emotions can be distinguished (Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger, Contempt, None, Uncertainty, Faceless).

For our problem, it is not necessary to differentiate such specific emotions, but rather it is sufficient to differentiate three types of categories: positive, negative and neutral. What we are really interested in is whether the action that the robot has decided to perform is well received or not by the user. For example, what does it mean to express happiness as a response to an action, under our criteria, we understand and classify the emotion happiness as a positive response. Following this criterion, we relabelled the AffectNet database and created an additional one in order to adjust the model to the user it is currently assisting. In this way, the pre-trained model is finely tuned to the expressions of the actual patient and his or her environment, making it more effective in the context of application. 

<p align="center">
<img src = "images/tabla_eng.JPG" width="700" />
</p>

# Table of contents

# Dependencies
To install the required packages, run pip install -r requirements.txt
# Project Structure

# Usage

# Datasets
- Own Dataset
https://drive.google.com/drive/folders/187Pg1hq5Bi1o-dYWYC47xSVxOLf8Pyte?usp=sharing
- AffectNet Dataset

# Data preparation

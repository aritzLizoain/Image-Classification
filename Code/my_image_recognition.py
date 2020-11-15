# -*- coding: utf-8 -*-
"""
//////////////////////////////////////////////////////////////////////////////////////////
// Original author(s): https://medium.com/intuitive-deep-learning/build-your-first-convolutional-neural-network-to-recognize-images-84b9c78fe0ce
// Modified by: Aritz Lizoain
// Github: https://github.com/aritzLizoain
// My personal website: https://aritzlizoain.github.io/
// Description: Image recognition with Keras (CIFAR-10 standard dataset)
// Copyright 2020, Aritz Lizoain.
// License: MIT License
//////////////////////////////////////////////////////////////////////////////////////////

Two trained models:
    *my_cifar10_model : original model. 77.25% accuracy on the test set.
    *my_cifar10_model2_augmented : model trained after applying data augmentation.
     78.04% accuracy on the test set.
1) Loading a trained model
2) Predicting on the test set
3) Evaluation of the predictions
4) Predicting on OWN IMAGES
"""

from keras.models import load_model #loading the model
from skimage.transform import resize #resize image
import numpy as np 
import matplotlib.pyplot as plt
import pickle #data loading
from sklearn.metrics import classification_report #classification report
import random

"""
1) Loading a trained model
"""

#model = load_model('Models/my_cifar10_model.h5') # original model
model = load_model('Models/my_cifar10_model2_augmented.h5') # +data augmentation model

"""
2) Predicting on the test set
"""

#Load the test set
x_test=pickle.load(open("Data/x_test.dat","rb"))
y_test=pickle.load(open("Data/y_test.dat","rb"))

y_test_label=np.argmax(np.round(y_test),axis=1)

number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',\
                   'frog', 'horse', 'ship', 'truck']

#Predict on the test set
predicted_classes=model.predict(x_test)   
#finding the positon (=label) of the prediction and solution
predicted_classes_label=np.argmax(np.round(predicted_classes),axis=1)

"""
3) Evaluation of the predictions
"""

#Comparing correct answers
correct=np.where(predicted_classes_label==y_test_label)[0] #need to have the same shape
print("Found", len(correct), "correct classes")   
#Comparing incorrect answers
incorrect=np.where(predicted_classes_label!=y_test_label)[0] #need shape
print("Found", len(incorrect), "incorrect classes")
#Visusalizing a random incorrect prediction
random = random.randint(0, len(incorrect))
plt.subplot(2,2,1)
plt.imshow(x_test[incorrect[random]].reshape(32,32,3),cmap='gray',interpolation='none')
plt.title('Predicted '+str(number_to_class[predicted_classes_label[incorrect[random]]])+\
          ', correct '+str(number_to_class[y_test_label[incorrect[random]]]))
plt.tight_layout()

#Classification report sklearn.metrics.classification_report
#Will help identifiying the misclassified classes in more detail. 
#Able to observe for which class the model performed better or worse.
target_names = [number_to_class[i] for i in range(10)]
print(classification_report(y_test_label, predicted_classes_label,\
                            target_names=target_names))              
#Recal:"how many of this class you find over the whole number of element of
# this class"
#Precision:"how many are correctly classified among that class"
#F1-score:the harmonic mean between precision & recall. Good on inbalanced sets, like this one
#Support:the number of occurence of the given class in your dataset

"""
4) Predicting on OWN IMAGES
*Prepare the image to recognize
*Predict on the image
*Analyze the prediction
"""

# PREPARE THE IMAGE TO RECOGNIZE
#reading the file as an array of pixel values
#the image is reshaped to (32,32,3)
#if the image is originally not squared, shape will be lost and recognition will be more likely to fail.
my_image=plt.imread("Images/my_image_1.jpg")
#resize image to fit model
#model image size: 32*32*3
my_image_resized=resize(my_image, (32,32,3)) #already makes values between 0-1
#visualize the image
img=plt.imshow(my_image) #will only show one
#img_resized=plt.imshow(my_image_resized)

# PREDICT ON THE IMAGE
probabilities=model.predict(np.array([my_image_resized]))
#np.array changes the current array of the pixel values into a 4D array
#because model.predict expects a 4D array (3D+training examples). 
#Training set and test set were consistent with this before
#10 output neurons corresponding to a probability distribution over the classes

# ANALYZE THE PREDICTION
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], "-- Probability:",\
      probabilities[0,index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:",\
      probabilities[0,index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:",\
      probabilities[0,index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:",\
      probabilities[0,index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:",\
      probabilities[0,index[5]])

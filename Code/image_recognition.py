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

1) Data processing: one-hot encoding and scaling
2) Building and training the CNN
3) Training the model
4) Model training process evaluation
5) Evaluation of the model
6) Saving the trained model
"""

from keras.datasets import cifar10 # CIFAR-10 dataset
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential #to set the CNN architecture
from keras.layers import Dense, Dropout, Flatten, Conv2D,\
    MaxPooling2D # NN layers
import pickle #to save the datasets
import random

"""
1) Data processing
*One-hot encoding labels keras.utils.to_categorical
*Scale image pixel values: change data type and divide by 255
"""

#Loading the dataset: CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#50000 training and 10000 testing samples.
#Training images: 32 pixels in height, 32 pixels in width, 3 pixels in depth
#Labels: 1 number (corresponding to the category) for each image. 



#Visualize a random image
random = random.randint(0, len(x_train))
img=plt.imshow(x_train[random])
print('The label of this category is: ',y_train[random])
#0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog,
#7 horse, 8 ship, 9 truck

#One-hot encoding conversion with Keras
y_train_one_hot=keras.utils.to_categorical(y_train,10)
y_test_one_hot=keras.utils.to_categorical(y_test,10)
print('the one hot label of the random image is:', y_train_one_hot[random])

#Pixel values take value between 0 and 255 (RGB scale) -> make them between 0 and 1
x_train=x_train.astype('float32') #convert the type to float32, which is 
#a datatype that can store values with decimal points. then divide by 255
x_test=x_test.astype('float32')
x_train=x_train/255
x_test=x_test/255

#I will save x_test and y_test in order to test the model the other file (my_image_recognition.py)
pickle.dump(x_test, open("Data/x_test.dat", "wb"))
pickle.dump(y_test_one_hot, open("Data/y_test.dat", "wb")) 
#BE CAREFUL WITH PATH AND OVERWRITING DATA

"""
2) Building and training the CNN
*Defining the CNN architecture with Keras Sequential model
*Compiling the model
"""

#ARCHITECTURE: (ConvX2, Max Pool, Dropout)X2, FC, Dropout, Fc, Softmax

model=Sequential() #create empty sequential model and then add layers

#Layer 1: conv layer, filter size 3X3, stride size 1 in both dimensions,
#depth 32. Padding same and activation relu will apply to all layers.
#we will use ReLU activation for all our layers, except for the last layer
#remember that ReLU doesn't map negative values (no negative values here)
#no specification of stride default setting = 1
#input shape needs to be specified, but not for the following layers
model.add(Conv2D(32,(3,3), activation='relu', padding='same',\
                 input_shape=(32,32,3)))

#Layer 2: conv layer, filter size 3X3, stride size 1 in both dimensions,
#depth 32. Padding same and activation relu will apply to all layers.
#we would need padding 1 to achieve the same width and height, but 
#we will use 'same' padding for all the conv layers, aka zero pad.
model.add(Conv2D(32,(3,3), activation='relu', padding='same'))
    
#Layer 3: max pooling layer, pool size 2X2, stride 2 in both dimensions,
#max pooling layer stride default given by pool size
model.add(MaxPooling2D(pool_size=(2,2)))

#Layer 4: dropout layer with probability 25% of dropout, to prevent overfitting
model.add(Dropout(0.25))

#Layers 5-8: same but depth of conv layer is 64 instead of 32
model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Layer 9: FC (Fully-Connected) layer. Now our neurons are not just in 
#on row, but spatially arranged in a cube-like format. We need to make
#them into one row, flattening. Flatten layer.
model.add(Flatten()) #now one row
model.add(Dense(512,activation='relu')) #dense (FC) layer. 

#Layer 10: another dropout of probability 50%
model.add(Dropout(0.5))

#Layers 11-12: another dense (FC) layer with 10 neurons and sofrmax activation
#last layer, softmax, only transforms the output of the previous layer 
#into probability distributions, which is the final goal
model.add(Dense(10,activation='softmax'))

#end of architecture
model.summary()

#COMPILING THE MODEL: model.compile
#First we need to specify with algorithm to use for the optimization, what loss 
#function to use and what other metrics to track apart from the loss function
#optimizer: adam. Adds some tweaks to stcochastic gradient 
#descent such that it reaches the lower loss function faster 
#loss: categorical crossentropy = loss function for classification.
#metrics: accuracy = we want to track accuracy on top of the loss function
model.compile(loss='categorical_crossentropy', optimizer='adam',\
              metrics=['accuracy'])

"""
3) Training the model
(highly time-consuming)
"""

#We are fitting the parameters to the data. We specify the data we are 
#training on, then the size of our mini-batch and how long we want to train it
#for. Last, specify the validation data, that will tell us how we are doing on
#the validation data at each point. We didn't split it before, we now specify
#how much of our dataset will be used as a validation set. In this case, 20% will be validation set.
hist=model.fit(x_train,y_train_one_hot, batch_size=32,epochs=20,\
               validation_split=0.2)
    
    
"""
4) Model training process evaluation
If the improvements in our model to the training set look matched up with the 
imporvements to the validation set, it doesn't seem like overfitting is a huge
problem in this model.
"""

# LOSS 
plt.plot(hist.history['loss']) #variable 1 to plot
plt.plot(hist.history['val_loss']) #variable 2 to plot
plt.title('Model loss') #title
plt.ylabel('Loss') #label y
plt.xlabel('Epoch') #label x
plt.legend(['Training', 'Validation'], loc='upper right') #legend
plt.show() #display the graph

# ACCURACY
plt.plot(hist.history['accuracy']) #variable 1 to plot
plt.plot(hist.history['val_accuracy']) #variable 2 to plot
plt.title('Model accuracy') #title
plt.ylabel('Accuracy') #label y
plt.xlabel('Epoch') #label x
plt.legend(['Training', 'Validation'], loc='lower right') #legend
plt.show() #display the graph

"""
5) Evaluation of the model (on the test set)
"""

print('The accuracy of the model on the test set is: ',\
      model.evaluate(x_test,y_test_one_hot)[1]*100,'%')
# The accuracy of the model on the test set is: 77.25%

"""
6) Saving the trained model
The model will be saved in a file format called HDF5
In order to load it run: 
    from keras.models import load_model
    model = load_model('my_cifar10_model.h5')
"""

model.save('model_name.h5') #be careful, don't overwrite
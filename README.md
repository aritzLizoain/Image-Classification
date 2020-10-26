# Image Classification

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorflow)
![GitHub last commit](https://img.shields.io/github/last-commit/aritzLizoain/Image-Classification)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/aritzLizoain/Image-Classification)
[![](https://tokei.rs/b1/github/aritzLizoain/Image-Classification?category=lines)](https://github.com/aritzLizoain/Image-Classification) 
![GitHub Repo stars](https://img.shields.io/github/stars/aritzLizoain/Image-Classification?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/aritzLizoain/Image-Classification?style=social)

Image recognition implementation with **Keras**. A **CNN** is built and trained with the **CIFAR-10** dataset. Two models are trained: one without data-augmentation (77.25% accuracy) and the other with data-augmentation (78.04% accuracy). Process:

``` image_recognition.py ```
* Data processing: one-hot encoding and scaling
* Building and training the CNN
* Training the model
* Training process evaluation
* Evaluation of the model
* Saving the trained model

``` my_image_recognition.py ```
* Loading the trained model
* Predicting on the test set
* Evaluation of the predictions
* Predicting on my own images

Followed [Course](https://medium.com/intuitive-deep-learning/build-your-first-convolutional-neural-network-to-recognize-images-84b9c78fe0ce)

## Predicting on my own images

Some are correct :heavy_check_mark: some are not :x:

<pre>
<img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/My_images/my_image_1.jpg" height="200"/>           <img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/Outputs/my_image_1_prediction.png" height="200"/> 
</pre>

<pre>
<img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/My_images/my_image_2.jpg" height="200"/>           <img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/Outputs/my_image_2_prediction.png" height="200"/> 
</pre>

<pre>
<img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/My_images/my_image_3.jpg" height="200"/>           <img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/Outputs/my_image_3_prediction.png" height="200"/> 
</pre>

<pre>
<img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/My_images/my_image_4.jpg" height="200"/>           <img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/Outputs/my_image_4_prediction.png" height="200"/> 
</pre>

<pre>
<img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/My_images/my_image_5.jpg" height="200"/>           <img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/Outputs/my_image_5_prediction.png" height="200"/> 
</pre>

<pre>
<img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/My_images/my_image_6.jpg" height="200"/>           <img src="https://github.com/aritzLizoain/Image-Classification/blob/main/Images/Outputs/my_image_6_prediction.png" height="200"/> 
</pre> 

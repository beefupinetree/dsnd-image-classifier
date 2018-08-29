# Image classifier - Flowers

This is the final project of the Deep learning section of the 'Data Scientist' nanodegree at Udacity. I import a deep convolutional neural network and add a custom classifier layer to the end. I then train the model on [this](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html "Flowers!") dataset which contains thousands of labeled images of flowers broken down into 102 categories.

The final product is a flexible convolutional neural network which can be called from the command line. It can learn from any set of labeled images and be used to label new images. The hyper-parameters can also be modified by the user from the command line itself. Finally the main model itself can be chosen by the user from all the possibilities within the TorchVision.Models library.

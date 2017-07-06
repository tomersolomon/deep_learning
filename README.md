## Tomer Solomon
## Foray into Deep Learning 

This is example of my work with neural networks, adapted from Andreas Mueller's Applied Machine Learning course (namley HW #5). Specifically, I used Keras Sequential interface (which uses Tensorflow as the backend). These tasks were run on an external GPU (specifically the Habanero computer cluster) because of the processing time recquired to determine the necessary parameters in a neural network, which was also a good experience for me. This was a great foray into seeing how powerful deep learning is and learning (no pun inteded) how to implement models in real life. 

THe assignments are organized via Task folders, and the prompt summaries are below. 

Homework #5: https://docs.google.com/document/d/1L-VoM88gJ1avThtdWuHHA3K8TpvvIn1jQiXZYt0UCSM/edit# 

### Task 1

A multilayer perceptron (feed forward neural network) with two hidden layers and rectified linear nonlinearities on the iris dataset. 

Test loss: 0.137
Test Accuracy: 0.974

### Task 2

A multilayer perceptron on the MNIST dataset. Compare a “vanilla” model with a model Qusing drop-out.

Vanilla Scores
Test loss: 0.108
Test Accuracy: 0.967

Dropout Scores
Test loss: 0.148
Test Accuracy: 0.957

### Task 3

A convolutional neural network on the SVHN dataset in format 2 (single digit classification) + a model that includes batch normalization.

Street View House Numbers Dataset: http://ufldl.stanford.edu/housenumbers/ 

Base Model
Test loss: 0.252
Test Accuracy: 0.927

Batch Normalization Model
Test loss: 0.134
Test Accuracy: 0.957


### Task 4

This was my favorite assignment by far. For this, we trained a model on the pets dataset (37 categories with 200 images per category) to classify different species of pets. The weighted of a pre-trained CNN (specifically VGG) were loaded and this CNN was used as feature extraction method to train a linear model. Access to the Habanero computer cluster closed but I ended up getting an accuracy of 85%.

Pet database: http://www.robots.ox.ac.uk/~vgg/data/pets/





# Deep Learning Nanodegree -- Lab 

## Motivation

This repo covers some of the Udacity's labs run during: 

* Artificial Intelligence Engineer Nanodegree (second term) []()
* and Deep Learning Nanodegree Foundations [Original Resource](http://github.com/udacity/deep-learning/)

## Index

### Batch Normalization Theory and Mini project with TensorFlow

* [Batch Normalization lesson](./batch_normalization/Batch_Normalization_Lesson.ipynb)

This Jupyter Notebook cover What is Batch Normalization, benefits, how to do it in TensorFlow, Graphical comparisons between identical networks, with and without batch normalization, brief explanation about the math behind Batch Normalization, and how to implementing manually using TensorFlow low abstraction API -tf.nn- instead of using higher abstraction API -tf.layers-, why the difference between training and inference. 

* [Exercises](./batch_normalization/Batch_Normalization.ipynb)

The original solution notebook provided by Udacity can be found [here](https://github.com/udacity/deep-learning/blob/master/batch-norm/Batch_Normalization_Solutions.ipynb). This notebook here, includes personal notes about the comparison and implementation of BN in TF using low abstraction level API in order to go through the theory exposed in the previous notebook in a deeper way. 


### Build your first models with Keras

* [Binary Classifier in Keras](./intro_to_keras/IMDB_in_Keras.ipynb)

With this project, you can do your first steps in training a feedforward network to predict sentiment analysis using IMDB reviews (positive or negative review). The project is quite simple which goal is to learn how to work with [datasets already in Keras](https://keras.io/datasets/), split your sets, one-hot encoding of the output, build your architecture, train, etc. 

* [Multiclass Classifier in Keras](./intro_to_keras/student_admissions.ipynb)

In this notebook, we predict student admissions to graduate school at UCLA based on three pieces of data: GRE Scores (Test), GPA Scores (Grades), Class rank (1-4). Again, the purpose of this notebook is to go through a multiclassification problem in Keras using an FF neural network. 


### Word Embeddings

* [Embeddings using Skip-gram](./embeddings/skip-gram_word2vec.ipynb)

When you're dealing with language and words, you end up with tens of thousands of classes to predict, one for each word. Trying to one-hot encode these words is massively inefficient, you'll have one element set to 1 and the other 50,000 set to 0. The word2vec algorithm finds much more efficient representations by finding vectors that represent the words. These vectors also contain semantic information about the words. Words that show up in similar contexts, such as "black", "white", and "red" will have vectors near each other. There are two architectures for implementing word2vec, CBOW (Continuous Bag-Of-Words) and Skip-gram. In this notebook, we implement the word2vec algorithm using the skip-gram architecture in TensorFlow. 


### Convolutions NN mini-labs

* [CNN MiniProjects](./dl--lab/aind2-cnn/)

This is an introducer mini lab for the [Dog Breed Classifier](https://github.com/nvmoyar/aind2-dog-breed-classifier). The idea is to learn how to do the tasks described below in Keras: 

* [Performance comparison of a perceptron classifier vs a cnn to classify CIFAR-10 images](./aind2-cnn/cifar10-classification/cifar10_mlp.ipynb) 
* [How to build a CNN in Keras](./aind2-cnn/cifar10-classification/cifar10_cnn.ipynb)
* [How to display the activation maps or filters in a convolution](./aind2-cnn/conv-visualization/conv_visualization.ipynb)
* [How to use Image augmentation](./aind2-cnn/cifar10-augmentation/cifar10_augmentation.ipynb)
* [How to apply Transfer-learning](./aind2-cnn/transfer-learning/)

### Semisupervised Learning

* (./semi_supervised_learning/semi_supervised_learning.ipynb)

This will be moved to an independent repo with more content related. This is a GAN project using SVHN dataset. Since this is a semisupervised problem, therefore we go through a problem where some of the data is labeled and some not. This is a very interesting and more realistic problem since it is closer to real life problems, where not all the information is labeled. 


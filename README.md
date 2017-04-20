# Introduction
This is assignment from Machine Learning course by coursera(Standford University)

## 1. Linear Regression

### 1.1 Linear Regression with one variable (ex1.m)
Predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities You would like to use this data to help you select which city to expand to next.

![1.1](/images/1.1.PNG)


The file ex1data1.txt contains the dataset for our linear regression problem. The first column is the population of a city and the second column is the profit of a food truck in that city. A negative value for profit indicates a loss.

### 1.2 Linear regression with multiple variables (ex1_multi.m)
Predict the prices of houses. Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to first collect information on recent houses sold and make a model of housing
prices.

The file ex1data2.txt contains a training set of housing prices in Portland, Oregon. The first column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house.


## 2. Logistic Regression

### 2.1 Logistic Regression (ex2.m)
Predict whether a student gets admitted into a university. Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams. You have historical data from previous applicants that you can use as a training set for logistic regression. For each training example, you have the applicant’s scores on two exams and the admissions decision.

![2.1](/images/2.1.png)

The task is to build a classification model that estimates an applicant’s probability of admission based the scores from those two exams.

### 2.2 Regularized logistic regression (ex2_reg.m)
Predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly. Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. From these two tests, you would like to determine whether the microchips should be accepted or
rejected. To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.

![2.2](/images/2.2.png)

## 3. Multi-class Classification and Neural Networks

### 3.1 Multi-class Classification (ex3.m)

Using logistic regression and neural networks to recognize handwritten digits (from 0 to 9). Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. This exercise will show you how the methods you’ve learned can be used for this classification task

![3.1](/images/3.1.png)

In the first part of the exercise, you will extend your previous implemention of logistic regression and apply it to one-vs-all classification

### 3.2 Neural Networks (ex3_nn.m)

In the previous part we use multi-class logistic regression to recognize handwritten digits. However, logistic regression cannot
form more complex hypotheses as it is only a linear classifier

In this part of the exercise, you will implement a neural network to recognize handwritten digits using the same training set as before. The neural network will be able to represent complex models that form non-linear hypotheses. For this week, you will be using parameters from a neural network that we have already trained. Your goal is to implement the feedforward propagation algorithm to use our weights for prediction. In next week’s exercise, you will write the backpropagation algorithm for learning the neural network parameters

![3.2](/images/3.2.png)

## 4. Neural Networks Learning

### 4.1 Backpropagation (ex4.m)

Implement feedforward propagation for neural networks and used it to predict handwritten digits with the weights we
provided. In this exercise, you will implement the backpropagation algorithm to learn the parameters for the neural network

### 4.2 Visualizing the hidden layer (ex4.m)
One way to understand what your neural network is learning is to visualize what the representations captured by the hidden units. To visualize the "representation" captured by the hidden unit is to reshape this 400 dimensional vector into a 20 × 20 image and display it

![4.2](/images/4.2.png)

## 5. Regularized Linear Regression and Bias v.s.Variance

### 5.1 Regularized Linear Regression (ex5.m)
Implement regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir

![5.1](/images/5.1.png)

### 5.2 Bias-variance(ex5.m)
An important concept in machine learning is the bias-variance tradeoff. Models with high bias are not complex enough for the data and tend to underfit, while models with high variance overfit to the training data.

In this part of the exercise, There are plot training and test errors on a learning curve to diagnose bias-variance problems.

![5.2](/images/5.2.png)

### 5.3 Polynomial regression (ex5.m)
The problem with our linear model was that it was too simple for the data and resulted in underfitting (high bias). In this part of the exercise, will address this problem by adding more features.

![5.3.1](/images/5.3.1.png)

![5.3.2](/images/5.3.2.png)

## 6. Support Vector Machines

### 6.1 Support Vector Machines (ex6.m)
In the first half of this exercise, will be using support vector machines (SVMs) with various example 2D datasets. Experimenting with these datasets will help gain an intuition of how SVMs work and how to use a Gaussian kernel with SVMs. In the next half of the exercise, will be using support vector machines to build a spam classifier.

![6.1.1](/images/6.1.1.png)

![6.1.2](/images/6.1.2.png)

![6.1.3](/images/6.1.3.png)

### 6.2 Spam Classification (ex6 spam.m)

Many email services today provide spam filters that are able to classify emails into spam and non-spam email with high accuracy. In this part of the exercise, will use SVMs to build your own spam filter.T raining a classifier to classify whether a given email, x, is
spam (y = 1) or non-spam (y = 0). 

In particular, you need to convert each email into a feature vector x 2 Rn. The following parts of the exercise will walk through how such  vector can be constructed from an email. 

![6.2](/images/6.2.png)

## 7.K-means Clustering and Principal Component Analysis

### 7.1 K-means Clustering (ex7.m)

Implement the K-means algorithm use for image compression. You will first start on an example 2D dataset that will help you gain an intuition of how the K-means algorithm works. After that, you wil use the K-means algorithm for image compression by reducing
the number of colors that occur in an image to only those that are most common in that image

![7.1.1](/images/7.1.1.png)

![7.1.2](/images/7.1.2.png)

### 7.2 Principal Component Analysis (ex7_pca.m)
Perform dimensionality reduction. You will first experiment with an example 2D dataset to get intuition on how PCA works, 

![7.2.1](/images/7.2.1.png)

![7.2.2](/images/7.2.2.png)


and then use it on a bigger dataset of 5000 face image dataset.

![7.2.3](/images/7.2.3.png)

## 8. Anomaly Detection and Recommender Systems

### 8.1 Anomaly detection (ex8.m)

Detecting anomalous behavior in server computers. The features measure the throughput (mb/s) and latency (ms) of response of each server. While your servers were operating, you collected m = 307 examples of how they were behaving, You suspect that the
vast majority of these examples are "normal" (non-anomalous) examples of the servers operating normally, but there might also be some examples of servers acting anomalously within this dataset.

![8.1](/images/8.1.png)

You will use a Gaussian model to detect anomalous examples in your dataset. You will first start on a 2D dataset that will allow you to visualize what the algorithm is doing. On that dataset you will fit a Gaussian distribution and then find values that have very low probability and hence can be considered anomalies. After that, you will apply the anomaly detection algorithm to a larger dataset with many dimensions.

### 8.2  Recommender Systems (ex8_cofi.m.)
Implement the collaborative filtering learning algorithm and apply it to a dataset of movie ratings.2 This dataset
consists of ratings on a scale of 1 to 5. The dataset has nu = 943 users, and nm = 1682 movies.

![8.2](/images/8.2.png)


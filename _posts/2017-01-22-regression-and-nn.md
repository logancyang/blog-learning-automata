---
toc: true
layout: post
description: Notes for Hinton's Neural Network course
categories: [note]
title: "Hinton Neural Network course"
---

## Regression

The word regression comes from "regression to the mean" like the parent-children height example. Later it
actually means *using functional form to approximate a bunch of data points*.

**i.i.d.**: Independent and identically distributed, a fundamental assumption for the data.

- Linear Regression
- Polynomial Regression
- Cross Validation
  - Split the training set into training and validation set. K-fold cross validation.
  - To determine the complexity of the model (degree of polynomial in the case of polynomial regression), choose the complexity with the lowest validation error

<br>
## Neural Networks

- Perceptron: $ \mathbf{wx} = y $, threshold $ y $ to get $ \hat{y} $
  - If linearly separable, it *will* find a solution
  - If not, it won't stop. But there's no way to know when to stop and declare it's not linearly separable.
- Gradient Descent: $ \mathbf{wx} = a $, activation. There's no thresholding.
- Perceptron vs. Gradient descent
  - Perceptron:
$$
  \Delta w_i = \eta * (y - \hat y) x_i
$$

  - Gradient descent:
$$
  \Delta w_i = \eta * (y - a) x_i
$$

  - Difference: activation with or without thresholding. Gradient descent is more robust, it needs a differentiable loss. The 1-0 loss of perceptron is not differentiable.
  - To get a differentiable / softer thresholding function, we have sigmoid (literally means s-like).
- Sigmoid unit in NN is just a softer thresholding version of Perceptron.
- Back-propagation for error: computationally beneficial organization of the chain rule.
- Local optima: For a single sigmoid unit, the error looks like a parabola because its quadratic. For many sigmoid units as in NN, there will be a lot of local optima.
- Learning = optimization
- Note that a 1-layer NN with sigmoid activation is equivalent to Logistic Regression!
- NN complexity
  - More nodes
  - More layers
  - Larger weights
- Restriction bias: what it is that you are able to represent, the restriction on possible hypotheses functions
  - Started out with Perceptron, linear
  - Then more perceptrons for more complex functions
  - Then use sigmoids instead of 1-0 hard thresholding
  - So NN doesn't have much restriction. It can represent any boolean function (more units), any continuous function (one hidden layer), and any arbitrary function with discontinuities (more hidden layers).
  - Danger of overfitting: use certain network structures, and cross validation.
- Preference bias: given two algorithm representations, why we prefer one over the other
  - Initialize weights at small random values
  - Always prefer simpler models when the error is similar: Occam's Razor
- Summary
  - Perceptrons: thresholding unit
  - Networks can produce any Boolean function.
  - Perceptron rule: finite time for linearly separable data
  - General differentiable rule - back propagation & gradient descent

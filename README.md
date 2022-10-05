# Neural Networks and Deep Learning Homeworks
***

In this repository you will find all the homeworks I did for the Neural Networks and Deep Learning course at University of Padua. In each of these homeworks 
we explored different machine learning techniques (supervised learning, unsupervised learning and reinforcement learning) by using neural networks. All the 
code was implemented using PyTorch.





## Homework 1: Supervised Learning (regression and classification)
In this work I implemented simple neural network models for solving two classical supervised learning tasks. The approximation of a function using a regression model and the classification of images from the Fashion MNIST dataset. For both tasks I explored the usage of different optimizers and regularization methods. The hyperparameters were optimised using a gridsearch for the regression and a Bayesian optimisation using Optuna for the classification task.
* The complete report of this homework can be found [**here**](https://github.com/hcapettini2/NNDL/blob/main/Homework_1_Supervised_Learning/Homework_1_Report_Capettini.pdf)
<p float="left">
  <img src="https://github.com/hcapettini2/NNDL/blob/main/Homework_1_Supervised_Learning/imgs/regression/fit.svg" type="image/svg+xml" width="500" />
</p>

## Homework 2: Unsupervised Learning (Autoencoder, Variational Autoencoder, Generative Adversarial Network)
In this work I explored different unsupervised learning techniques based on deep learning. First a convolutional autoencoder was trained and tested on the fashion MNIST dataset, its learning parameters were optimized using OPTUNA. Then the best convolutional autoencoder was used in the context of transfer learning to perform a supervised classification task. Finally a variational convolutional autoencoder and a Generative Adversarial Network were trained and tested with the purpose of generating new samples.
* The complete report of this homework can be found [**here**](https://github.com/hcapettini2/NNDL/blob/main/Homework_2_Unsupervised_Learning/Homework_2_Report_Capettini.pdf)
<p float="left">
  <img src="https://github.com/hcapettini2/NNDL/blob/main/Homework_2_Unsupervised_Learning/Img/GAN.gif" width="500" />
</p>


## Homework 3: Reinforcement Learning (Q-Learning)
In this work we explored how to use neural networks to solve reinforcement learning problems under the paradigm of Deep Q-Learning. A learning agent was trained to solve the CartPole-v1 environment using the state representations provided by the environment. Also a learning agent was implemented to solve the LunarLander-v2 environment directly from the state representation.
* The complete report of this homework can be found [**here**](https://github.com/hcapettini2/NNDL/blob/main/Homework_3_Reinforcement_Learning/Homework_3_Report_Capettini.pdf)
<p float="left">
  <img src="https://github.com/hcapettini2/NNDL/blob/main/Homework_3_Reinforcement_Learning/Images/Pole_Gifs/Trained_Pole_0.gif" width="500" />
  <img src="https://github.com/hcapettini2/NNDL/blob/main/Homework_3_Reinforcement_Learning/Images/Lander_Gifs/Trained_Lander_3.gif" width="500" />
</p>

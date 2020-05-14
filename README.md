# Hand-Written-Digit-Recognition
This repository contains my contribution towards the completion of Hand written Digit Recognition using the MNIST dataset.

This project implements back propagation algorithm to recognize hand written digits that are part of the MNIST dataset. 

On running the file mlp.py  the output shows 3,000 samples in training, 10,000 samples in validation and 10,000 samples in testing.

The file activation.py contains the function sigmoid(z) which returns the value of the sigmoid function and sigmoid_prime(z) that returns the 
value of the derivative.

The file bc.py contains the back propagation algorithm. 
Gradient check can be used to check if the implementation is correct or not. The gradients for several weights are to be checked before training 
the network, by changing the values of layer_id, unit_id and weight_id in mlp.py

The final predictions for the test datatset are stored in a CSV file in one-hot encoding format.

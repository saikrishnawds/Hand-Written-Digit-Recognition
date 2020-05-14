import matplotlib.pyplot as plt 
import numpy as np 
import math

def sigmoid(z):
    
    
    y = 1/(1 + np.exp(-z)) 
 
    return y

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    
    y = 1/(1 + np.exp(-z)) 
    dz=y*(1-y)
  
    
    return dz


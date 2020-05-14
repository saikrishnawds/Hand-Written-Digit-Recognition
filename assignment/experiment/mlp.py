#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')


    return parser.parse_args()


def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)


def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784,30,10])
    # train the network using SGD
    evaluation_cost,evaluation_accuracy,training_cost,training_accuracy,epochs=model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=128,
        eta=1e-3,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    prediction=[np.argmax(model.feedforward(test_data[0][x])) for x in range(len(test_data[0]))]
    accuracy=model.accuracy(test_data,convert='False')
  
    ohe=[network2.vectorized_result(i) for i in prediction]
    with open('predictions.csv','w',newline='') as csvfile:
        record=csv.writer(csvfile)
        for i in ohe:
            record.writerow(int(j) for j in (i))
    plt.figure()
    plt.plot(range(epochs),evaluation_cost,range(epochs),training_cost)
    plt.legend(('evaluation_cost','traning_cost'),loc='upper right')
    plt.title('loss trend')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.show()
    plt.figure()
    plt.plot(range(epochs),evaluation_cost,range(epochs),training_cost)
    plt.legend(('evaluation_accuracy','traning_accuracy'),loc='upper right')
    plt.title('accuracy trend')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
   
    #Testing the trained model
    n=len(test_data[0])
    accuracy=model.accuracy(test_data)
    print("[Testing Accuracy]: {} / {}".format(accuracy,n))
    results=[np.argmax(model.feedforward(test_data[0][x])) for x in range(len(test_data[0]))]
    results=np.array(results)
    m=results.max() +1
    results=np.eye(m)[results]
    #print(results)
    np.savetxt("predictions.csv",results,fmt="%d",delimiter=",") # using %d to convert the floating point values to int and store in the csv file
    
    
    
    
if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()


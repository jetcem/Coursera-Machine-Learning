# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 03:00:01 2020

@author: Cem
"""

import numpy as np
import matplotlib.pyplot as plt


def plotting_data(X,m):
    visual = np.random.permutation(range(m)) #shuffle the data index to plot mixed data
    Plot_data = X.reshape(X.shape[0],20,20)  #reshape the data to be able to show as image of 20x20
    Plot_data = np.moveaxis(Plot_data, 1, 2)
    
    columns = 4
    rows = 5
    
    fig=plt.figure(figsize=(8, 8))
    
    for i in range(1, columns*rows +1):
        img = Plot_data[visual[i]]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

def plotting_NN(T):
    Plot_data = T[:,1:].reshape(T.shape[0],20,20)
    Plot_data = np.moveaxis(Plot_data, 1, 2)
    
    columns = 5
    rows = 5
    
    fig=plt.figure(figsize=(8, 8))
    
    for i in range(1, columns*rows +1):
        img = Plot_data[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()
    
def sigmoid(x):                 #sigmoid function
   return 1 / (1 + np.exp(-x))

def relu(x):
   return np.maximum(0,x)

def sigmoid_prime(y):           #derivative of sigmoid
    s = 1 / (1 + np.exp(-y))
    y = s*(1-s)
    return y

def randInitializeWeights(L_in,L_out):  #randomly initializing first weights before we train our model
    W = np.zeros((L_out, 1 + L_in))
    # epsilon_init = (6**1/2) / (L_in + L_out)**1/2
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in)*(2*epsilon_init)-epsilon_init
    return W

def predict(T1,T2,X):
    X_R = np.insert(X,0,1.0,axis=1)
    z2 = np.matmul(X_R,T1.transpose())
    a2 = sigmoid(z2)
    a2 = np.insert(a2,0,1.0,axis=1)
    z3 = np.matmul(a2,T2.transpose())
    a3 = sigmoid(z3)

    max_index_col = (np.argmax(a3, axis=1))+1
    
    return max_index_col



def nnCostFunction(T1,T2,input_layer_size,hidden_layer_size,num_labels,X, y, lambda_rate):
    m = np.size(X,0) 
    J = 0
    X = np.insert(X,0,1.0,axis=1)   #adding bias to input layer
    v = np.zeros((m,num_labels))    #create zeros for output unit as 10 class classification
    Theta1 = T1.copy()
    Theta2 = T2.copy()

    for i in range(m):              #define output units variable
        v[i][(y[i])-1] = 1
        
    z2 = np.matmul(X,Theta1.transpose())
    a2 = sigmoid(z2)
    a2 = np.insert(a2,0,1.0,axis=1)
    z3 = np.matmul(a2,Theta2.transpose())
    a3 = sigmoid(z3)
    #Non-Regularized Cost
    for i in range(num_labels):
        J_test = (1/m)*(np.matmul(-v[:,i].transpose(),np.log(a3)[:,i])-np.matmul(1-v[:,i].transpose(),np.log(1-a3)[:,i]))
        J = J+J_test
        
    #Regularized Cost    
    Regular_part = (lambda_rate/(2*m))*(sum(sum(np.power(Theta1[:,1:],2)))+sum(sum(np.power(Theta2[:,1:],2))))
    J = J+Regular_part
       
    #Back Propagation
    der_error_3 = (a3 - v)
    der_error_2 = np.multiply(np.matmul(der_error_3,Theta2[:,1:]),sigmoid_prime(z2))
    delta_2 = np.matmul(der_error_3.transpose(),a2)
    delta_1 = np.matmul(der_error_2.transpose(),X)

    Theta1[:,0] = 0 #setting first column of theta1 to zeros for regularization of backpropagation algorithm
    Theta2[:,0] = 0 #setting first column of theta2 to zeros for regularization of backpropagation algorithm

    Theta1_grad_reg = (1/m)*(delta_1+lambda_rate*Theta1);
    Theta2_grad_reg = (1/m)*(delta_2+lambda_rate*Theta2);

    return J,Theta1_grad_reg,Theta2_grad_reg


def GradientNN(X,y,Theta1,Theta2,alpha,num_iters,lambda_rate,input_layer_size, hidden_layer_size, num_labels):
    J_cost = np.zeros((num_iters,1))
    for i in range(num_iters):
        cost, grad1, grad2 = nnCostFunction(Theta1,Theta2,input_layer_size, hidden_layer_size, num_labels,X, y,lambda_rate)
        Theta1 = Theta1 - (alpha * grad1)
        Theta2 = Theta2 - (alpha * grad2)
        J_cost[i] = cost
        print("Cost= {} , Iteration = {}".format(cost,i))

    return Theta1 , Theta2,cost,J_cost
    
    
    
    
    
    
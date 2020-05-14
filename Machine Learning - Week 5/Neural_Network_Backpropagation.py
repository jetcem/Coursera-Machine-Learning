"""
Created on Sun Apr 26 03:01:15 2020

@author: Cem Dogan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import io
import functions


#=========== Part 1: Loading and Visualizing Data =============
mat = io.loadmat('data\ex4data1.mat') #read the data from mat file
data_X = mat["X"]   
data_y = mat["y"]

rand = np.random.permutation(data_X.shape[0])   #create random perm. index list to shuffle the data
data_X = np.take(data_X,rand,axis=0,out=data_X)
data_y = np.take(data_y,rand,axis=0,out=data_y)
X_train= data_X[500:]   #split the data as train,test
y_train=data_y[500:]
X_test  = data_X[:500]
y_test = data_y[:500]

# Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
m = np.size(X_train,0)          # (note that we have mapped "0" to label 10)

print('Loading and visualizing the data...\n')
functions.plotting_data(X_train,m)
input('Program paused. Press enter to continue.\n')

#================ Part 2: Loading Parameters ================
print('Loading Saved Neural Network Parameters ...\n')
mat_param = io.loadmat('data\ex4weights.mat') #These are the initial parameters provided by coursera to validate our code
initial_theta1 = mat_param['Theta1']
initial_theta2 = mat_param['Theta2']


#================ Part 3: Compute Cost (Feedforward) ================
print('Feedforward Using Neural Network ...\n')
lambda_rate = 0
J,Theta_1,Theta_2 = functions.nnCostFunction(initial_theta1,initial_theta2,input_layer_size,hidden_layer_size,num_labels,X_train, y_train, lambda_rate)
print("Cost at parameters (loaded from ex4weights): {} \n(this value should be approximately about 0.287629)\n".format(J))
input('Program paused. Press enter to continue.\n')
# NOTE : We divided the data as train and test. This cost values are only acceptable for non divided data. Thats why it says approximately

#=============== Part 4: Implement Regularization ===============
print('Checking Cost Function (w/ Regularization) ... \n')
lambda_rate = 1
J,Theta_1,Theta_2 = functions.nnCostFunction(initial_theta1,initial_theta2,input_layer_size,hidden_layer_size,num_labels,X_train, y_train, lambda_rate)
print("Cost at parameters (loaded from ex4weights): {} \n(this value should be approximately about 0.383770)\n".format(J))
input('Program paused. Press enter to continue.\n')
# NOTE : We divided the data as train and test. This cost values are only acceptable for non divided data. Thats why it says approximately

# ================ Part 6: Initializing Pameters ================
initial_theta1 = functions.randInitializeWeights(input_layer_size, hidden_layer_size)
initial_theta2 = functions.randInitializeWeights(hidden_layer_size, num_labels)

#=============== Part 7: Implement Backpropagation ===============
lambda_rate = 1
grad_alpha = 2
num_iters = 500
Theta_1, Theta_2,J,Cost_curve = functions.GradientNN(X_train,y_train,initial_theta1,initial_theta2,grad_alpha,num_iters,lambda_rate,input_layer_size, hidden_layer_size, num_labels)

print('\nCost Function Visualization ...')
curve_iter = np.linspace(0,num_iters,num_iters)
fig=plt.figure(figsize=(6, 6))
plt.plot(curve_iter,Cost_curve[:,0])
plt.xlabel("Number of Iterations")
plt.ylabel("J(\u03F4)")
plt.title("Cost Function")
plt.show()


print('\nNeural Network Visualization ...')
functions.plotting_NN(Theta_1)

pred = functions.predict(Theta_1, Theta_2, X_train).reshape(y_train.shape[0],1)
accuracy = np.mean((pred == y_train)*1)*100
print("\nTraining Set Accuracy: {} \n".format(accuracy))

pred_test = functions.predict(Theta_1, Theta_2, X_test).reshape(y_test.shape[0],1)
accuracy_test = np.mean((pred_test == y_test)*1)*100
print("\nTest Set Accuracy: {} \n".format(accuracy_test))

input('Program paused. Press enter to continue random prediction stage.\n')


s = ''
for i in range(1,X_test.shape[0]):
    rand = np.random.permutation(range(X_test.shape[0]))    #shuffle the data to create random visualization
    pred_data = X_test.reshape(X_test.shape[0],20,20)       # reshape the data to show the images
    pred_data = np.moveaxis(pred_data, 1, 2)                #images are in reversed position. To fix this, move the axis
    img = pred_data[rand[i]]                                #randomly show the images
    plt.imshow(img)
    plt.show()
    
    X_p = X_test[rand[i]].reshape(1,X_test.shape[1])        #X_p has (400,) to fix this : rehape the X_p as(1,400)
    pred_ = functions.predict(Theta_1, Theta_2, X_p)
    print("'\nNeural Network Prediction: {} , (digit {})".format(pred_,pred_%10))
    
    s = input('Paused - press enter to continue, q to exit:');
    if s == 'q':
      break


data={'Theta1': Theta_1, 'Theta2': Theta_2}
io.savemat('test.mat', data, oned_as='row')













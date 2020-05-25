# -*- coding: utf-8 -*-
"""
Created on Sun May 24 20:17:58 2020

@author: Cem
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

def mapFeature(degree,X):
    feature_degree = degree
    new_features = np.ones((X.shape[0],sum(range(degree + 2)))) #creating the new_features array
    column = 1
    for i in range(1,feature_degree+1): #this loop is for calculating the new columns values
            for j in range(i+1):
                new_features[:,[column]] = (np.power(X[:,0],(i-j))*np.power(X[:,1],(j)))[:,np.newaxis] 
                column +=1
    return new_features   

def sigmoid(x):
  return 1 / (1 + np.exp(-x)) #sigmoid function for cost and gradient calculation

def predict(theta,X):         #predict whether the y value > 0.5 or not   
    m = X.shape[0]
    predicted = np.zeros((m,1))
    predicted = (sigmoid(np.matmul(X,theta))>=0.5)*1
    probability = sigmoid(np.matmul(X,theta))
    return predicted,probability

def Cost_Grad(new_features,y,iteration_num = 1000,learning_rate=0.1,alpha = 1):
    initial_theta = np.ones(new_features.shape[1]) #it is just created for final theta values and assigned to theta for first creation
    theta = np.array(initial_theta).reshape(initial_theta.shape[0],1)
    y = y.reshape(y.shape[0],1) #I had a problem with array shapes I did this to change shape from (118,) to (118,1)

    hypothesis = sigmoid(new_features.dot(theta)); #calculate the hypothesis
    m = new_features.shape[0]                      #total number of examples 
    J_Cost = 0
    X = new_features

    for i in range(iteration_num):
        hypothesis = sigmoid(np.matmul(X,theta))                            #calculate the hypothesis
        a = np.matmul(-y.transpose(),np.log(hypothesis))                    #calculate the cost function's first part
        b = np.matmul((1-y).transpose(),np.log(1-hypothesis))               #calculate the cost function's second part
        J_Cost = (1/m)*(a-b) + (alpha/(2*m)) * sum(np.power(theta[1:],2));  #calculate the cost
        d_0 = (1/m)*(X[:,0].transpose().dot((hypothesis-y)))                #calculate the x0 part of the gradient
        d_all = (1/m)*np.matmul(X[:,1:].transpose(),(hypothesis-y))         #calculate the x1:end part of the gradient
        #d_0 is the part which has to be non-regularized gradient section
        theta[0]   = theta[0]- learning_rate*d_0                            #this is the non-regularized part of the gradient descent
        theta[1:]  = theta[1:]*(1-(learning_rate*alpha/m))- learning_rate*d_all     #this is the regularization part of the gradient descent    
    return theta

def plt_boundary(theta,X,y,feature_map_degree):
    
    pos = np.where(y==1)        #find the indexes of y where y is positive
    neg = np.where(y==0)        #find the indexes of y where y is negative
    
    p1 = plt.plot(X[pos,0],X[pos,1],marker = '+',markersize = 9,color = 'k')[0] #create a plot for y==1 and y==0
    p2 = plt.plot(X[neg,0],X[neg,1],marker = 'o',markersize = 7,color = 'y')[0]
    
    u = np.linspace(-1, 1.5, 50) #this is a kind of meshgrid. We should create this to catch all points in our decision boundary
    v = np.linspace(-1, 1.5, 50)
        
    z = np.zeros(( len(u), len(v) )) 

    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.matmul(mapFeature(feature_map_degree,np.array([u[i],v[j]]).reshape(1,2)),theta)
            #As we already calculated the theta. We can now create a boundary with theta values on meshgrid
            #z is the height values over the contour. It will create 50,50 for our plot
            #with this line we actually create another dimension from 2D data to 3D
            #the values which calculated with theta will be circled 
    z = np.transpose(z)
    
    p3 = plt.contour(u, v, z, levels=[0], linewidth=2).collections[0]
    plt.legend((p1,p2, p3),('y = 1', 'y = 0', 'Decision Boundary'), numpoints=1, handlelength=0)

    plt.show(block=False)
    
    """
    you can change the levels parameter in plt.contour to see how 
    you can plot decision boundaries with respect to the z value
    it specifies the height. If you choose 0 it will plot the line
    under the height 0 which created by z.

    It is complicated for beginning, I even miss the meaning, sorry for the poor exp.

    Lets try again:
    So if you create mesh for 50,50 points in the border of -1 to 1.5 
    and then you multiply every point with our theta array
    this will create a kind of probability for our point as we do that process
    in the predict section. For example if output is 0.3 then identify
    predicted output as 0,if output is 0.7(>0.5) then identify predicted output as 1.
    So; after defining every points probability in the 50x50 meshgrid(plot area)
    we choose the points which has probability above the 0.
    Under the zero probabilities means they are out of the hypothesis area
    if you write to command window below code:
    z[25,25]
    it will give you that points probability:
    1.3956860767785348
    
    So open the plot, the point of z[25,25] is approximately
    middle point of the plot.
    Which means most of the values in this area will be determined as
    y==1. That is why we see the probability > 0 .
    We can get the probability of every point in 50x50 grid(mesh)
    with this approachwe only plot the area which has probability above 0
    """
    
    """
    this is an optional plot to understand the data in 3d
    z matrix is creating a kind of height for x and y axes
    after we multiply the mesh with theta 
    we reach a kind of height of probabilities
    after we create another dimension we can recognize y==1 or y==0
    with respect to the height, we can choose the data
    """
    
    """
    u,v = np.meshgrid(np.linspace(-1, 1.5, 50),np.linspace(-1, 1.5, 50))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(u,v,z,cmap='Dark2',vmin=0.5, vmax=1,antialiased=True )
    # ax.scatter3D(u,v,z)
    fig.show()
    
    e = np.linspace(-1, 1, 453)
    plt.scatter(z[z>0.5],e)

"""
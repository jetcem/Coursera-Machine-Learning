# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 00:38:01 2020

@author: Cem
"""

"""
Regularized LogisticRegression
"""
import numpy as np
import pandas as pd

import Functions as rf


df = pd.read_csv('ex2data2.txt')    #import the data 
y = np.array(df.iloc[:,2])          #split the data for y(output)
X = np.array(df.iloc[:,0:2])        #split the data for training features

feature_map_degree = 6              #define the feature map degree

new_features = rf.mapFeature(feature_map_degree,X)  #call the feature map function
    
theta = rf.Cost_Grad(new_features,y,5000,0.1,alpha=1) #call the cost and gradient function

# hypothesis = rf.sigmoid(np.matmul(new_features,theta))  
    
rf.plt_boundary(theta,X,y,feature_map_degree)  #call the plotting function which is for decision boundary

predicted_data,probability = rf.predict(theta,new_features) #predict the y values for training set
accuracy = np.mean((predicted_data == y[:,np.newaxis])*1)   #calculate the accuracy
print('Train accuracy of the model = %',accuracy*100)


from sklearn.linear_model import LogisticRegression    # this part is optional, it is imported just to check accuracy
y = np.ravel(y)
loj_model = LogisticRegression(solver = 'liblinear').fit(new_features,y)
loj_model.intercept_
loj_model.coef_ #bağımsız değişkenlere ilişkin katsayılar (ağırlıklar)

y_pred = loj_model.predict(new_features)
solver_accuracy = np.mean((y_pred == y)*1)
print('Train accuracy of the sklearn = %',solver_accuracy*100)
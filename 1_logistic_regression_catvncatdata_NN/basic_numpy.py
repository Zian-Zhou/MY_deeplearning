# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 09:10:34 2018

@author: a
"""

import math 
import numpy as np
#=================================1 - Building basic functions with numpy

#----------------------------Sigmoid函数及其导函数

def basic_sigmoid(x):
    s = 1.0/(1+1/math.exp(x))
    return s
"""
basic_sigmoid(3)
"""
#做不了向量式计算

def sigmoid(x):
    s = 1.0/(1+1/np.exp(-x))
    return s
"""example
x = np.array([1, 2, 3])
sigmoid(x)
"""
def sigmoid_derivative(x):
    s = 1.0/(1+1/np.exp(-x))
    ds = s * (1-s)
    return ds

#-----------------------------Reshaping arrays
#- X.shape is used to get the shape (dimension) of a matrix/vector X. 
#- X.reshape(…) is used to reshape X into some other dimension.
#For example, if you would like to reshape an array v of shape (a, b, c) 
#into a vector of shape (a*b,c) you would do:  
    #v = v.reshape((v.shape[0]*v.shape[1], v.shape[2])) # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
    
def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2]),1)
    return v
"""example
This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image)))
"""                 

#-------------------------Normalizing rows
#Another common technique we use in Machine Learning and Deep Learning 
#is to normalize our data. It often leads to a better performance because 
#gradient descent converges faster after normalization.     

#归一化
def normalizeRows(x):
    # 二范数：Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    # ||x||_2
    x_norm = np.linalg.norm(x,axis = 1,keepdims = True)
    # x = x/||x||_2
    x = x/x_norm #利用numpy的广播，用矩阵与列向量相除
    return x
"""
x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))
#normalizeRows(x) = [[ 0.          0.6         0.8       ]
# [ 0.13736056  0.82416338  0.54944226]]
"""

#------------------------Broadcasting and the softmax function
#More Details Broadcasting:https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
def softmax(x):
    """Calculates the softmax for each row of the input x.
    Argument:
    x -- A numpy matrix of shape (n,m)
    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp,axis = 1,keepdims = True)
    s = x_exp / x_sum #gbroadcasting
    return s

"""
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))
"""  
  
###NOTE
"""
What you need to remember: 
- np.exp(x) works for any np.array x and applies the exponential function to every coordinate 
- the sigmoid function and its gradient 
- image2vector is commonly used in deep learning 
- np.reshape is widely used. In the future, you’ll see that keeping your matrix/vector dimensions straight will go toward eliminating a lot of bugs. 
- numpy has efficient built-in functions 
- broadcasting is extremely useful
"""
                    
#=======================================2.Vectorization  

#--------------------------Implement the L1 and L2 loss functions
# GRADED FUNCTION: L1

def L1(yhat, y):
    loss = np.sum(np.abs(y-yhat))
    return loss
"""
yhat = np.array([.2, 0.2, 0.2, .2, .2])
y = np.array([0, 0, 0, 0, 0])
print("L1 = " + str(L1(yhat,y)))
"""                
      
def L2(yhat,y):
    loss = np.sum( np.power((y-yhat),2))
    #np.power(x,m):x.^m
    return loss
"""
yhat = np.array([.2, .2, .2, .2, .2])
y = np.array([0, 0, 0, 0, 0])
print("L2 = " + str(L2(yhat,y)))
"""                  



    
    
    
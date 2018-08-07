# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:00:12 2018

@author: a
"""
################Face Recognition for the Happy House & Art: Neural Style Transfer#####

#%%=========================Part 2：Deep Learning & Art: Neural Style Transfer
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import cv2
%matplotlib inline

#%%------------1 - Problem Statement
"""
Neural Style Transfer (NST) is one of the most fun techniques in deep learning. 
As seen below, it merges two images, namely, a “content” image (C) and a “style” image (S), 
to create a “generated” image (G). 
The generated image G combines the “content” of the image C with the “style” of image S.
"""
#%%------------2 - Transfer Learning
"""
Following the original NST paper (https://arxiv.org/abs/1508.06576), we will use the VGG network. 
Specifically, we’ll use VGG-19, a 19-layer version of the VGG network. 
This model has already been trained on the very large ImageNet database, 
and thus has learned to recognize a variety of low level features (at the earlier layers) and high level features (at the deeper layers).
"""
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)

"""
The model is stored in a python dictionary where each variable name is the key and the corresponding value is a tensor containing that variable’s value. 
To run an image through this network, you just have to feed the image to the model. 
In TensorFlow, you can do so using the tf.assign function. 
In particular, you will use the assign function like this:
    model["input"].assign(image)
This assigns the image as an input to the model. 
After this, if you want to access the activations of a particular layer, say layer 4_2 when the network is run on this image, you would run a TensorFlow session on the correct tensor conv4_2, as follows:
    sess.run(model["conv4_2"])
"""

#%%----------3 - Neural Style Transfer
"""
We will build the NST algorithm in three steps:

Build the content cost function Jcontent(C,G)
Build the style cost function Jstyle(S,G)
Put it together to get J(G)=αJcontent(C,G)+βJstyle(S,G).
"""


#3.1 - Computing the content cost
content_image = scipy.misc.imread("images/claude-monet.jpg")
imshow(content_image)

"""
the total idea is that:
    * 3.1.1 - How do you ensure the generated image G matches the content of the image C?*
"""
#Compute the “content cost” using TensorFlow.
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_C_unrolled = tf.reshape(a_C,[n_H*n_W,n_C])
    a_G_unrolled = tf.reshape(a_G,[n_H*n_W,n_C])
    
    J_content = 1./(4*n_H*n_W*n_C)*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))
    
    return J_content

"""
tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))
"""

"""
What you should remember: 
- The content cost takes a hidden layer activation of the neural network, and measures how different a(C) and a(G) are. 
- When we minimize the content cost later, this will help make sure G has similar content as C.
"""

#%%       3.2 - Computing the style cost
style_image = scipy.misc.imread("images/claude-monet.jpg")
imshow(style_image)


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A,tf.transpose(A))
    
    return GA

"""
tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)

    print("GA = " + str(GA.eval()))
"""

#3.2.2 - Style cost
def compute_layer_style_cost(a_S,a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    m,n_H,n_W,n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S,[n_H*n_W,n_C]))
    a_G = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))
    
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    
    J_style_layer = 1./(4*n_C*n_C*n_H*n_H*n_W*n_W) * tf.reduce_sum(tf.square(tf.subtract(GS,GG)))
    
    return J_style_layer
"""
tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)

    print("J_style_layer = " + str(J_style_layer.eval()))
"""

#%%   3.2.3 Style Weights
"""
So far you have captured the style from only one layer.
 We’ll get better results if we “merge” style costs from several different layers. 
 After completing this exercise, feel free to come back and experiment with different weights to see how it changes the generated image GG.
 But for now, this is a pretty reasonable default:
"""
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model,STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2):
        J_style(S,G) = sum_l(lambda_l * Jstyle_l(S,G))  that Jstyle_l(S,G) is given by one layer(l)
    """
    
    J_style = 0
    
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]# Select the output tensor of the currently selected layer
        a_S = sess.run(out)# Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        
        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out####

        J_style_layer = compute_layer_style_cost(a_S,a_G)
        
        J_style += coeff * J_style_layer

    return J_style
#Note: In the inner-loop of the for-loop above, a_G is a tensor and hasn’t been evaluated yet. It will be evaluated and updated at each iteration when we run the TensorFlow graph in model_nn() below.

"""
What you should remember: 
- The style of an image can be represented using the Gram matrix of a hidden layer’s activations.
  However, we get even better results combining this representation from multiple different layers. 
  This is in contrast to the content representation, where usually using just a single hidden layer is sufficient. 
- Minimizing the style cost will cause the image G to follow the style of the image S. 
"""


#%%3.3 - Defining the total cost to optimize
def total_cost(J_content,J_style,alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    
    return J
"""
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()    
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))
"""

#%%-----------------------------------4 - Solving the optimization problem
"""
You’ve previously implemented the overall cost J(G). 
We’ll now set up TensorFlow to optimize this with respect to G. 
To do so, your program has to reset the graph and use an “Interactive Session“. 
Unlike a regular session, the “Interactive Session” installs itself as the default session to build a graph.
This allows you to run variables without constantly needing to refer to the session object, which simplifies the code.
"""
# `````````````Reset the graph
tf.reset_default_graph()
# ````````````Start interactive session
sess = tf.InteractiveSession()


###````````````load image:content,style
#content_image = scipy.misc.imread("images/louvre_small.jpg")
#content_image = reshape_and_normalize_image(content_image)
#style_image = scipy.misc.imread("images/monet.jpg")
#style_image = reshape_and_normalize_image(style_image)
content_image = scipy.misc.imread("images/my_content.jpg")
content_image = cv2.resize(content_image,(400,300),interpolation=cv2.INTER_CUBIC)
content_image = reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread("images/my_style.jpg")
style_image = cv2.resize(style_image,(400,300),interpolation=cv2.INTER_CUBIC)
style_image = reshape_and_normalize_image(style_image)


"""
Go to “/images” and upload your images (requirement: (WIDTH = 400, HEIGHT = 300)) 400*300, rename them “my_content.png” and “my_style.png” for example.
Change the code in part (3.4) from :
"""
###`````````````create G image
generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

###````````````load VGGnet model which have been trained well
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

###````````````compute content cost
"""
To get the program to compute the content cost, we will now assign a_C and a_G to be the appropriate hidden layer activations.
 We will use layer conv4_2 to compute the content cost.
 The code below does the following:
     1.Assign the content image to be the input to the VGG model.
     2.Set a_C to be the tensor giving the hidden layer activation for layer “conv4_2”.
     3.Set a_G to be the tensor giving the hidden layer activation for the same layer.
     4.Compute the content cost using a_C and a_G.
"""
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out####
J_content = compute_content_cost(a_C, a_G)
#Note: At this point, a_G is a tensor and hasn’t been evaluated. 
#      It will be evaluated and updated at each iteration when we run the Tensorflow graph in model_nn() below.


###`````````````compute style cost
sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)

###``````````````compute total cost
J = total_cost(J_content, J_style, 1, 3)

###`````````````optimizer
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)

###````````````path
output_path = "images/3/"


###````````````model
def model_nn(sess,input_image,path,num_iterations=200):
    sess.run(tf.global_variables_initializer())
    
    sess.run(model["input"].assign(input_image))
    
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model["input"])
        
        if i%2==0:
            print("Iteration = "+ str(i))
        if i%20==0:
            Jt,Jc,Js = sess.run([J,J_content,J_style])
            print("Iteration "+str(i)+":")
            print("total cost = "+str(Jt))
            print("content cost = "+str(Jc))
            print("style cost = "+str(Js))
            
            save_image(path+str(i)+".png",generated_image)
            
    save_image(path+'generated_image.jpg',generated_image)

    return generated_image

#run the model
model_nn(sess, generated_image,output_path)








# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 20:03:07 2018

@author: a
"""
#%%==================================Part 1：Keras tutorial - the Happy House
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline

#%%---------------------------------1.The Happy House
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

"""
number of training examples = 600
number of test examples = 150
X_train shape: (600, 64, 64, 3)
Y_train shape: (600, 1)
X_test shape: (150, 64, 64, 3)
Y_test shape: (150, 1)
"""
#%%--------------------------2 - Building a model in Keras
"""An example of a model in Keras!
def model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32,(7,7),strides = (1,1),name = 'conv0')(X)
    X = BatchNormalization(axis = 3,name = 'b0')(X)
    X = Activation('relu')(X)
    
    #Maxpool
    X = MaxPooling2D((2,2),name='max_pool')(X)
    
    ## FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(input = X_input,outputs = X,name='HappyModel')

    return model
"""
def HappyModel(input_shape):
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((3,3))(X_input)
    
    X = Conv2D(32,(7,7),strides = (1,1),name = 'conv0')(X)
    X = BatchNormalization(axis = 3,name = 'bn0')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,2),name='max_pool')(X)
    
    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X)
    
    model = Model(input = X_input,outputs = X,name = 'HappyModel')
    
    return model

"""
You have now built a function to describe your model. To train and test this model, there are four steps in Keras: 
1. Create the model by calling the function above 
2. Compile the model by calling model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"]) 
3. Train the model on train data by calling model.fit(x = ..., y = ..., epochs = ..., batch_size = ...) 
4. Test the model on test data by calling model.evaluate(x = ..., y = ...)
"""

#step1 : create the model
happyModel = HappyModel(X_train.shape[1:])

#step2 : compile the model to configure the learning process
happyModel.compile(optimizer = "Adam",loss = "binary_crossentropy",metrics = ["accuracy"])

#step3 : train the model
happyModel.fit(x = X_train, y = Y_train, epochs = 10, batch_size = 32)
"""note
Note that if you run fit() again, 
the model will continue to train with  
the parameters it has already learnt 
instead of reinitializing them.

my question:how can i get the parameters(weights,biases,and so on),or at the next time,i should use keras again
"""
#step4 : test/evaluate the model
preds = happyModel.evaluate(X_test,Y_test)
print()
print("Loss = "+str(preds[0]))
print("Test Acuuracy = "+str(preds[1]))

#%%%------------------other useful function
#
happyModel.summary()

#
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))



"""
What we would like you to remember from this assignment: 
- Keras is a tool we recommend for rapid prototyping. 
  It allows you to quickly try out different model architectures. 
  Are there any applications of deep learning to your daily life that you’d like to implement using Keras? 
- Remember how to code a model in Keras and the four steps leading to the evaluation of your model on the test set. 
  Create->Compile->Fit/Train->Evaluate/Test.
"""
#%%------------------------------Test on own image
img_path = 'images/ZLH.jpg'
### END CODE HERE ###
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))

#%%=================================Part 2：Residual Networks
"""
 In theory, very deep networks can represent very complex functions; 
 but in practice, they are hard to train. 
 Residual Networks, introduced by He et al., 
 allow you to train much deeper networks than were previously practically feasible
"""
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
%matplotlib inline

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
#%%---------1 - The problem of very deep neural networks

#%%---------2 - Building a Residual Network
"""
shortcut/skip connection
"""
#2.1 - The identity block
def identity_block(X,f,filters,stage,block):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'bn'+str(stage)+block+'_branch'
    
    F1,F2,F3 = filters
    X_shortcut = X
    
    #main path
    X = Conv2D(filters = F1,kernel_size = (1,1), strides = (1,1), padding = 'valid',
               name = conv_name_base+'2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2,kernel_size = (f,f),strides = (1,1), padding = 'same',
               name = conv_name_base+'2b',kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F3,kernel_size = (1,1),strides = (1,1),padding = 'valid',
               name = conv_name_base+'2c',kernel_initializer = glorot_uniform(seed=0))(X)
    
    #add shortcut value to main path,and then activate it
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X
    
tf.reset_default_graph()
with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float",[3,4,4,6])
    X = np.random.randn(3,4,4,6)
    A = identity_block(A_prev,f=2,filters = [2,4,6],stage = 1,block='a')
    test.run(tf.global_variables_initializer())
    out = test.run([A],{A_prev:X,K.learning_phase():0})
    print("out = "+str(out[0][1][1][0]))
    
#%%-----------------2.2 - The convolutional block
"""
前面是直接将最前层的值A传到末层，add之后再激活：这条路径称为shortcut path
现在The convolutional block就是在shortcut path路径中加入卷积层
- Its main role is to just apply a (learned) linear function that reduces the dimension of the input, 
  so that the dimensions match up for the later addition step.
"""
def convolutional_block(X,f,filters,stage,block,s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'bn'+str(stage)+block+'_branch'
    
    F1,F2,F3 = filters
    X_shortcut = X
    
    ###main path
    X = Conv2D(F1,(1,1),strides=(s,s),name = conv_name_base+'2a',kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(F2,(f,f),strides=(1,1),name = conv_name_base+'2b',padding = 'same',kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(F3,(1,1),strides=(1,1),name = conv_name_base+'2c',kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,name = bn_name_base+'2c')(X)

    ###shortcut path
    X_shortcut = Conv2D(F3,(1,1),strides=(s,s),name = conv_name_base+'1',kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base+'1')(X_shortcut)

    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    return X

tf.reset_default_graph()
with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float",[3,4,4,6])
    X = np.random.randn(3,4,4,6)
    A = convolutional_block(A_prev,f=2,filters = [2,4,6],stage = 1 ,block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A],{A_prev:X,K.learning_phase():0})
    print("out = "+str(out[0][1][1][0]))
 

#%%----------------------3 - Building your first ResNet model (50 layers)
"""Constructure:(50layers)
input->>Zero Pad->>stage1->>stage2->>stage3->>stage4->>stage5->>AVG POOL->>Flatten->>FC->>output

Zero Pad:pad of (3,3)

stage1:CONV(64fliters,(7,7),strides=(2,2),'conv1')->Batch Norm->Relu->MAX POOL((3,3),strides=(2,2))

stage2:CONV Block(3 set of filters[F1,F2,F3],size[64,64,256],s=1,f=3,block='a')->ID Block(3 set of filters[F1,F2,F3],size[64,64,256],f=3,block='b','c')*2
    
stage3:CONV Block(3 set of filters[F1,F2,F3],size[128,128,512],s=2,f=3,block='a')->ID Block(3 set of filters[F1,F2,F3],size[128,128,512],f=3,block='b','c','d')*3
    
stage4:CONV Block(3 set of filters[F1,F2,F3],size[256,256,1024],s=2,f=3,block='a')->ID Block(3 set of filters[F1,F2,F3],size[256,256,1024,f=3,block='b''c''d''e''f'])*5
    
stage5:CONV Block(3 set of filters[F1,F2,F3],size[512,512,2048],s=2,f=3,block='a')->ID Block(3 set of filters[F1,F2,F3],size[512,512,2048,f=3,block='b''c'])*2

AVG POOL:(2,2)
    
Flatten:The flatten doesn’t have any hyperparameters or name. 
    
FC:reduces its input to the number of classes using a softmax activation. Its name should be 'fc' + str(classes).
"""
def ResNet50(input_shape = (64,64,3), classes = 6):
    """
    Returns:
    model -- a Model() instance in Keras
    """
    X_input = Input(input_shape)
    
    X =  ZeroPadding2D((3,3))(X_input)
    
    ####stage1:CONV(64fliters,(7,7),strides=(2,2),'conv1')->Batch Norm->Relu->MAX POOL((3,3),strides=(2,2))
    X = Conv2D(64,(7,7),strides=(2,2),name='conv1',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)
    
    ####stage2:CONV Block(3 set of filters[F1,F2,F3],size[64,64,256],s=1,f=3,block='a')->ID Block(3 set of filters[F1,F2,F3],size[64,64,256],f=3,block='b','c')*2
    X = convolutional_block(X,f=3,filters=[64,64,256],stage=2,block='a',s=1)
    X = identity_block(X,3,[64,64,256],stage=2,block='b')
    X = identity_block(X,3,[64,64,256],stage=2,block='c')
    
    ####stage3:CONV Block(3 set of filters[F1,F2,F3],size[128,128,512],s=2,f=3,block='a')->ID Block(3 set of filters[F1,F2,F3],size[128,128,512],f=3,block='b','c','d')*3
    X = convolutional_block(X,f=3,filters=[128,128,512],stage=3,block='a',s=2)
    X = identity_block(X,3,[128,128,512],stage=3,block='b')
    X = identity_block(X,3,[128,128,512],stage=3,block='c')
    X = identity_block(X,3,[128,128,512],stage=3,block='d')
    
    ####stage4:CONV Block(3 set of filters[F1,F2,F3],size[256,256,1024],s=2,f=3,block='a')->ID Block(3 set of filters[F1,F2,F3],size[256,256,1024,f=3,block='b''c''d''e''f'])*5
    X = convolutional_block(X,f=3,filters=[256,256,1024],stage=4,block='a',s=2)
    X = identity_block(X,3,[256,256,1024],stage=4,block='b')
    X = identity_block(X,3,[256,256,1024],stage=4,block='c')
    X = identity_block(X,3,[256,256,1024],stage=4,block='d')
    X = identity_block(X,3,[256,256,1024],stage=4,block='e')
    X = identity_block(X,3,[256,256,1024],stage=4,block='f')
    
    ####stage5:CONV Block(3 set of filters[F1,F2,F3],size[512,512,2048],s=2,f=3,block='a')->ID Block(3 set of filters[F1,F2,F3],size[512,512,2048,f=3,block='b''c'])*2
    X = convolutional_block(X,f=3,filters=[512,512,2048],stage=5,block='a',s=2)
    X = identity_block(X,3,[512,512,2048],stage=5,block='b')
    X = identity_block(X,3,[512,512,2048],stage=5,block='c')

    ####AVG POOL:(2,2)
    X = AveragePooling2D((2,2),name='avg_pool')(X)
    
    ####output layer
    X = Flatten()(X)
    X = Dense(classes,activation='softmax',name='fc'+str(classes),kernel_initializer=glorot_uniform(seed=0))(X)
    
    model = Model(inputs = X_input,outputs = X,name='ResNet50')

    return model
#%%
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#-----------------------------------------------------------
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
#--------------------------------------------------------
model.fit(X_train, Y_train, epochs = 2, batch_size = 32)
"""
Epoch 1/2
1080/1080 [==============================] - 202s 187ms/step - loss: 2.8990 - acc: 0.3287
Epoch 2/2
1080/1080 [==============================] - 174s 161ms/step - loss: 2.1205 - acc: 0.4213
"""
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
"""
120/120 [==============================] - 6s 49ms/step
Loss = 13.4317459742
Test Accuracy = 0.166666667163
"""
#%%using the model have been trained,which has been trained on GPU
model = load_model('ResNet50.h5') 
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
#Test Accuracy = 0.866666662693
#%%test my own image
img_path = 'images/sign4_own.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)
my_image = scipy.misc.imread(img_path)
imshow(my_image)
print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
print(model.predict(x))
#%%
model.summary()
#%%
plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))













"""
What you should remember: 
- Very deep “plain” networks don’t work in practice because they are hard to train due to vanishing gradients. 
- The skip-connections help to address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn an identity function. 
- There are two main type of blocks: The identity block and the convolutional block. 
- Very deep Residual Networks are built by stacking these blocks together.
"""
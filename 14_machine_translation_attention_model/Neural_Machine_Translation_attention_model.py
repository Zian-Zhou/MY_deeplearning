# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 19:01:21 2018

@author: a
"""

#%%=======================================Part 1 - Neural Machine Translation(NMT), using an attention model
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
%matplotlib inline


#%%```````````````````````1 - Translating human readable dates into machine readable dates
"""
The model you will build here could be used to translate from one language to another, such as translating from English to Hindi. 
However, language translation requires massive datasets and usually takes days of training on GPUs. 
To give you a place to experiment with these models even without using massive datasets, 
we will instead use a simpler “date translation” task.

The network will input a date written in a variety of possible formats 
(e.g. “the 29th of August 1958”, “03/30/1968”, “24 JUNE 1987”) 
and translate them into standardized, machine readable dates 
(e.g. “1958-08-29”, “1968-03-30”, “1987-06-24”). 
We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD.
"""
#--------------------1.1 - Dataset
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
"""
- dataset: a list of tuples of (human readable date, machine readable date) 
- human_vocab: a python dictionary mapping all characters used in the human readable dates to an integer-valued index 
- machine_vocab: a python dictionary mapping all characters used in machine readable dates to an integer-valued index. These indices are not necessarily consistent with human_vocab. 
- inv_machine_vocab: the inverse dictionary of machine_vocab, mapping from indices back to characters.
"""
#dataset[:10]

"""
Tx=30 (which we assume is the maximum length of the human readable date; 
if we get a longer input, we would have to truncate it) 
and Ty=10 (since “YYYY-MM-DD” is 10 characters long).
"""
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)
"""
- X: a processed version of the human readable dates in the training set, where each character is replaced by an index mapped to the character via human_vocab. Each date is further padded to TxTx values with a special character (< pad >). X.shape = (m, Tx) 
- Y: a processed version of the machine readable dates in the training set, where each character is replaced by the index it is mapped to in machine_vocab. You should have Y.shape = (m, Ty). 
- Xoh: one-hot version of X, the “1” entry’s index is mapped to the character thanks to human_vocab. Xoh.shape = (m, Tx, len(human_vocab)) 
- Yoh: one-hot version of Y, the “1” entry’s index is mapped to the character thanks to machine_vocab. Yoh.shape = (m, Tx, len(machine_vocab)). Here, len(machine_vocab) = 11 since there are 11 characters (‘-’ as well as 0-9).
"""
index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])

#%%``````````````````````````````````2 - Neural machine translation with attention
"""
The attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step.
简而言之，翻译的时候不是看完一整段再翻译，而是先阅读一部分，翻译，再继续..
"""

#------------------------------------2.1 - Attention mechanism
#neural machine translation with attention.png
# Defined shared layers as global variables

"""
https://blog.csdn.net/pengjian444/article/details/56316445
"""
repeator = RepeatVector(Tx)#RepeatVector层将输入重复Tx次
concatenator = Concatenate(axis=-1)#该层接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量。
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)#计算两个tensor中样本的张量乘积。例如，如果两个张量a和b的shape都为（batch_size, n），则输出为形如（batch_size,1）的张量，结果张量每个batch的数据都是a[i,:]和b[i,:]的矩阵（向量）点积。

def one_step_attention(a,s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    s_prev = repeator(s_prev)
    
    # Use concatenator to concatenate a and s_prev on the last axis
    concat = concatenator([a, s_prev])
    
    ## Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
    e = densor1(concat)
    
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.    
    energies = densor2(e)
    
    # Use "activator" on "energies" to compute the attention weights "alphas"
    alphas = activator(energies)
    
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell
    context = dotor([alphas,a])

    return context

n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)



def model(Tx,Ty,n_a,n_s,human_vocab_size,machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    X = Input(shape=(Tx,human_vocab_size))
    s0 = Input(shape=(n_s,),name='s0')
    c0 = Input(shape=(n_s,),name='c0')
    s = s0
    c = c0
    
    outputs = []
    
    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True.
    a = Bidirectional(LSTM(n_a,return_sequences=True))(X)
    
    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t
        context = one_step_attention(a,s)
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state]
        s, _,  c = post_activation_LSTM_cell(context, initial_state = [s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM
        out = output_layer(s)
        
        # Step 2.D: Append "out" to the "outputs" list
        outputs.append(out)
        
    # Step 3: Create model instance taking three inputs and returning the list of outputs.
    model = Model(inputs=[X,s0,c0],outputs=outputs)
    
    return model

model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()


#As usual, after creating your model in Keras, you need to compile it and define what loss, optimizer and metrics your are want to use
"""
Compile your model using categorical_crossentropy loss, 
a custom Adam optimizer (learning rate = 0.005, β1=0.9, β2=0.999, decay = 0.01) and ['accuracy'] metrics:
"""
opt = Adam(lr=0.005,beta_1=0.9,beta_2=0.999,decay=0.01)
model.compile(loss = 'categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#The last step is to define all your inputs and outputs to fit the model
s0 = np.zeros((m,n_s))
c0 = np.zeros((m,n_s))
outputs = list(Yoh.swapaxes(0,1))

#%%FIT MODEL

model.fit([Xoh,s0,c0],outputs,epochs=1,batch_size=100)



#%%after trained(pre-trained weights)
model.load_weights('models/model.h5')


"""#examples
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:

    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]

    print("source:", example)
    print("output:", ''.join(output))
"""


#%%----------------------------3 - Visualizing Attention (Optional / Ungraded)
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64)





"""
Here’s what you should remember from this notebook:

Machine translation models can be used to map from one sequence to another. They are useful not just for translating human languages (like French->English) but also for tasks like date format translation.
An attention mechanism allows a network to focus on the most relevant parts of the input when producing a specific part of the output.
A network using an attention mechanism can translate from inputs of length Tx to outputs of length Ty, where Tx and Ty can be different.
You can visualize attention weights α⟨t,t′⟩ to see what the network is paying attention to while generating each output.
"""






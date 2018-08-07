# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 23:14:14 2018

@author: a
"""
#%%==================Part.1 Building your Recurrent Neural Network - Step by Step
import numpy as np
from rnn_utils import *

#%%``````````````1 - Forward propagation for the basic Recurrent Neural Network
"""
T_x=T_y
"""
#1.1 - RNN cell
def rnn_cell_forward(xt,a_prev,parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    a_next = np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba)
    yt_pred = softmax(np.dot(Wya,a_next)+by)
    
    cache = (a_next,a_prev,xt,parameters)
    
    return a_next,yt_pred,cache
"""
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", a_next.shape)
print("yt_pred[1] =", yt_pred[1])
print("yt_pred.shape = ", yt_pred.shape)
"""

#%%1.2 - RNN forward pass
def rnn_forward(x,a0,parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    caches = []
    
    n_x,m,T_x = x.shape
    n_y,n_a = parameters["Wya"].shape
    
    a = np.zeros((n_a,m,T_x))
    y_pred = np.zeros((n_y,m,T_x))
    
    a_next = a0
    
    for t in range(T_x):
        a_next,yt_pred,cache = rnn_cell_forward(x[:,:,t],a_next,parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
        
    caches = (caches,x)
    
    return a,y_pred,caches

"""
np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a, y_pred, caches = rnn_forward(x, a0, parameters)
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)
print("y_pred[1][3] =", y_pred[1][3])
print("y_pred.shape = ", y_pred.shape)
print("caches[1][1][3] =", caches[1][1][3])
print("len(caches) = ", len(caches))
"""

#%%2 - Long Short-Term Memory (LSTM) network
#2.1 - LSTM cell
def lstm_cell_forward(xt,a_prev,c_prev,parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wu -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bu -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    """
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wu = parameters["Wu"]
    bu = parameters["bu"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]
    
    n_x,m = xt.shape
    n_y,n_a = Wy.shape
    
    concat = np.zeros((n_a+n_x,m))
    concat[:n_a,:] = a_prev
    concat[n_a:,:] = xt
    
    forget_gate = sigmoid(np.dot(Wf,concat)+bf)
    update_gate = sigmoid(np.dot(Wu,concat)+bu)
    cct = np.tanh(np.dot(Wc,concat)+bc)
    c_next = forget_gate*c_prev + update_gate*cct
    output_gate = sigmoid(np.dot(Wo,concat)+bo)
    a_next = output_gate*np.tanh(c_next)
    yt_pred = softmax(np.dot(Wy,a_next)+by)
    
    cache = (a_next,c_next,a_prev,c_prev,forget_gate,update_gate,cct,output_gate,xt,parameters)
    
    return a_next,c_next,yt_pred,cache
"""
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
c_prev = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wu = np.random.randn(5, 5+3)
bu = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", c_next.shape)
print("c_next[2] = ", c_next[2])
print("c_next.shape = ", c_next.shape)
print("yt[1] =", yt[1])
print("yt.shape = ", yt.shape)
print("cache[1][3] =", cache[1][3])
print("len(cache) = ", len(cache))
"""

def lstm_forward(x,a0,parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """
    caches = []
    
    n_x,m,T_x = x.shape
    n_y,n_a = parameters["Wy"].shape
    
    a = np.zeros((n_a,m,T_x))
    c = np.zeros((n_a,m,T_x))
    y = np.zeros((n_y,m,T_x))
    
    a_next = a0
    c_next = np.zeros((n_a,m))
    
    for t in range(T_x):
        a_next,c_next,yt,cache = lstm_cell_forward(x[:,:,t],a_next,c_next,parameters)
        a[:,:,t] = a_next
        y[:,:,t] = yt
        c[:,:,t] = c_next
        caches.append(cache)
        
    caches = (caches,x)
    
    return a,y,c,caches
"""
np.random.seed(1)
x = np.random.randn(3,10,7)
a0 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wu = np.random.randn(5, 5+3)
bu = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)
print("a[4][3][6] = ", a[4][3][6])
print("a.shape = ", a.shape)
print("y[1][4][3] =", y[1][4][3])
print("y.shape = ", y.shape)
print("caches[1][1[1]] =", caches[1][1][1])
print("c[1][2][1]", c[1][2][1])
print("len(caches) = ", len(caches))
"""    
    
#%%3 - Backpropagation in recurrent neural networks (OPTIONAL / UNGRADED)
#3.1 - Basic RNN backward pass    
def rnn_cell_backward(da_next,cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """
    (a_next, a_prev, xt, parameters) = cache
    
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    dtanh = (1-a_next**2)*da_next
    
    dxt = np.dot(Wax.T,dtanh)
    dWax = np.dot(dtanh,xt.T)
    
    da_prev = np.dot(Waa.T,dtanh)
    dWaa = np.dot(dtanh,a_prev.T)
    
    dba = np.sum(dtanh,keepdims=True,axis=-1)
    
    gradients = {"dxt":dxt,"da_prev":da_prev,"dWax":dWax,"dWaa":dWaa,"dba":dba}
    
    return gradients
"""
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
b = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)

da_next = np.random.randn(5,10)
gradients = rnn_cell_backward(da_next, cache)
print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients["dba"][4])
print("gradients[\"dba\"].shape =", gradients["dba"].shape)
"""

def rnn_backward(da,caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """
    (caches,x) = caches
    (a1,a0,x1,parameters) = caches[0]
    
    n_a,m,T_x = da.shape
    n_x,m = x1.shape
    
    dx = np.zeros((n_x,m,T_x))
    dWax = np.zeros((n_a,n_x))
    dWaa = np.zeros((n_a,n_a))
    dba = np.zeros((n_a,1))
    da0 = np.zeros((n_a,m))
    da_prevt = np.zeros((n_a,m))
    
    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:,:,t]+da_prevt,caches[t])#da[:,:,t]+da_prevt: da的更新值
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"],gradients["da_prev"],gradients["dWax"],gradients["dWaa"],gradients["dba"]
        dx[:,:,t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        
    da0 = da_prevt
    
    gradients = {"dx":dx,"da0":da0,"dWax":dWax,"dWaa":dWaa,"dba":dba}

    return gradients
"""
np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
a, y, caches = rnn_forward(x, a0, parameters)
da = np.random.randn(5, 10, 4)
gradients = rnn_backward(da, caches)

print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients["dba"][4])
print("gradients[\"dba\"].shape =", gradients["dba"].shape)
"""

#%%3.2 - LSTM backward pass
def lstm_cell_backward(da_next,dc_next,cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWu -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbu -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """
    (a_next,c_next,a_prev,c_prev,ft,ut,cct,ot,xt,parameters) = cache
    
    n_x,m = xt.shape
    n_a,m = a_next.shape
    
    #derivate of gate:outputgate,cc,update gate,forget gate
    """
    按照自己手写的式子，下面四句不应该有后面一个乘项的，如ft*(1-ft)，
    但为了计算成本以及代码精简的目的，专门写成如下的形式，这样在之后用到时，不需要再重复计算同一次乘法
    """
    dot = (da_next*np.tanh(c_next)) * ot*(1-ot)
    dcct = (dc_next*ut + da_next*ot*(1-np.square(np.tanh(c_next)))*ut) * (1-np.square(cct))
    dut = (dc_next*cct + da_next*ot*(1-np.square(np.tanh(c_next)))*cct) * ut*(1-ut)
    dft = (dc_next*c_prev + da_next*ot*(1-np.square(np.tanh(c_next)))*c_prev) * ft*(1-ft)
    
    
#    dot = da_next * np.tanh(c_next) * ot * (1-ot)
#    dcct = (dc_next*it+ot*(1-np.square(np.tanh(c_next)))*it*da_next)*(1-np.square(cct))
#    dit = (dc_next*cct+ot*(1-np.square(np.tanh(c_next)))*cct*da_next)*it*(1-it)
#    dft = (dc_next*c_prev+ot*(1-np.square(np.tanh(c_next)))*c_prev*da_next)*ft*(1-ft)
    
    
    concat = np.concatenate((a_prev,xt),axis=0)#按行堆叠（上下合并）
    
    #derivate of parameters:Wf,bf,Wu,bu,Wo,bo,Wc,bc,a_prev,xt,c_prev
    dWf = np.dot(dft,concat.T)
    dbf = np.sum(dft,axis=1,keepdims=True)
    dWu = np.dot(dut,concat.T)
    dbu = np.sum(dut,axis=1,keepdims=True)
    dWo = np.dot(dot,concat.T)
    dbo = np.sum(dot,axis=1,keepdims=True)
    dWc = np.dot(dcct,concat.T)
    dbc = np.sum(dcct,axis=1,keepdims=True)
    
    da_prev =  np.dot(parameters["Wf"][:,:n_a].T,dft) + np.dot(parameters["Wu"][:,:n_a].T,dut) + np.dot(parameters["Wo"][:,:n_a].T,dot) + np.dot(parameters["Wc"][:,:n_a].T,dcct)
    dxt     =  np.dot(parameters["Wf"][:,n_a:].T,dft) + np.dot(parameters["Wu"][:,n_a:].T,dut) + np.dot(parameters["Wo"][:,n_a:].T,dot) + np.dot(parameters["Wc"][:,n_a:].T,dcct)
    dc_prev = dc_next*ft + da_next*ot*(1-np.square(c_next))*ft
    """
    1、parameters["Wf"]: 因为Wf = [Wfa,Wfx]，Wfa是Wf的前n_a列，Wfx是Wf的后n_x列
    2、注意顺序：求导后是Wfa.T * dft  （这里*是矩阵计算）
    """
    
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWu": dWu,"dbu": dbu,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    
    return gradients
    
"""
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
c_prev = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wu = np.random.randn(5, 5+3)
bu = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

da_next = np.random.randn(5,10)
dc_next = np.random.randn(5,10)
gradients = lstm_cell_backward(da_next, dc_next, cache)
print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])##
print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)##
print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])##
print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)##
print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])##
print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)##
print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])##
print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)##
print("gradients[\"dWu\"][1][2] =", gradients["dWu"][1][2])
print("gradients[\"dWu\"].shape =", gradients["dWu"].shape)
print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])##
print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)##
print("gradients[\"dbf\"][4] =", gradients["dbf"][4])##
print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)##
print("gradients[\"dbu\"][4] =", gradients["dbu"][4])
print("gradients[\"dbu\"].shape =", gradients["dbu"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])##
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)##
"""
    
def lstm_backward(da,caches):
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWu -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbu -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """
    (caches, x) = caches    
    (a1, c1, a0, c0, f1, u1, cc1, o1, x1, parameters) = caches[0]
    
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a+n_x))
    dWu = np.zeros((n_a, n_a+n_x))
    dWc = np.zeros((n_a, n_a+n_x))
    dWo = np.zeros((n_a, n_a+n_x))
    dbf = np.zeros((n_a, 1))
    dbu = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))
    
    for t in reversed(range(T_x)):
        gradients = lstm_cell_backward(da[:,:,t]+da_prevt,dc_prevt,caches[t])
        dx[:,:,t] = gradients['dxt']
        dWf = dWf + gradients['dWf']
        dWu = dWu + gradients['dWu']
        dWc = dWc + gradients['dWc']
        dWo = dWo + gradients['dWo']
        dbf = dbf + gradients['dbf']
        dbu = dbu + gradients['dbu']
        dbc = dbc + gradients['dbc']
        dbo = dbo + gradients['dbo']
    
    da0 = gradients['da_prev']
    
    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWu": dWu,"dbu": dbu,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    
    return gradients
    
    
"""
np.random.seed(1)
x = np.random.randn(3,10,7)
a0 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wu = np.random.randn(5, 5+3)
bu = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)

parameters = {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)

da = np.random.randn(5, 10, 4)
gradients = lstm_backward(da, caches)

print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
print("gradients[\"dWu\"][1][2] =", gradients["dWu"][1][2])
print("gradients[\"dWu\"].shape =", gradients["dWu"].shape)
print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
print("gradients[\"dbu\"][4] =", gradients["dbu"][4])
print("gradients[\"dbu\"].shape =", gradients["dbu"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)
"""    
   
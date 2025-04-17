from __future__ import division
import numpy as np
import torch.nn.functional as F
import pymc as pm
import pytensor.tensor as tt
#import theano.tensor as tt
import torch
from torch import FloatTensor

'''
Defino las funciones de activacion.
'''

#Semilla aleatoria para mayor solidez en la funcion noise relu.
np.random.seed()

def sigmoide(tensor,a):
    '''
    Funcion Sigmoide.
    :Entrada= Debe ser un "numpy array"
    '''
    #return 1.0/(1.0 + np.exp(-tensor))
    return (F.sigmoid(tensor))


def tan_h(tensor,a):
    '''
    Función tangente hiperbolica
    :Entrada= Debe ser un "numpy array"
    
    '''
    #return np.tanh(tensor)
    return (F.tanh(tensor))


def ReLU(tensor,a):
    '''
    Funcion Rectified Linear Unit
    :Entrada = Debe ser un "numpy array"
    
    '''
    #return np.maximum(0.0,tensor)
    #re = nn.ReLU(inplace=True)
    return (F.relu(tensor))
    #return (re(tensor))


def Noisy_ReLU(tensor,a):
    '''
    Nosiy Rectified Linear Unit. T.
    :tensor: numpy array
    :regresa: numpy array de elementos rectificados.
    '''
    return np.maximum(0.0,tensor+np.random.normal(0.0,1.0))

def Leaky_ReLU(tensor,a):
    '''
    Leaky Rectified Linear Unit.
    :tensor: numpy array
    :regresa: numpy array .
    '''
    #tensor[tensor<0] *= 0.01
    #return tensor
    #tensor = torch.tensor(tensor)
    return (F.leaky_relu(tensor))

def softmax(tensor,a):
    '''
    Funcion Softmax, similara a la funcion exponencial normalizada: Entrega un vector de probabilides de acierto que sumados dan 1.
    :Entrada: Valores en 1D y del tipo numpy array.
    
    '''
    #tensor = tensor -np.max(tensor, axis=1,keepdims=True)
    #exp = np.exp(tensor)
    return (F.softmax(tensor, dim=1)) #exp/np.sum(exp, axis=1,keepdims=True)

def softmax_Bay(tensor,a):
    '''
    Funcion Softmax bayesiana-debe señalar el vector de probabilidades a prior: Entrega un vector de probabilides de acierto,sumados dan 1.
    :Entrada: Valores en 1D y del tipo numpy array.
    
    '''
    if (a==0):
        #print(tensor)
        observed_soft = F.softmax(tensor, dim=1)
        observed_soft = observed_soft.data.numpy()
        tensor = tensor.data.numpy()
        row_obser , col_obser = observed_soft.shape
        prob = np.asarray([[0.60, 0.40, 0.30, 0.23, 0.43,0.34, 0.45, 0.62, 0.78 , 0.24]]) #Probabilidades propuestas por clase
        prob = np.repeat(prob,64,axis=0)
        
        with pm.Model() as Sof:
            # Defino priors
            proba = pm.Dirichlet('proba', a=prob, shape=(row_obser,col_obser))
            z = pm.Normal('z',mu=tensor, sigma=1.5, shape=(row_obser,col_obser))
        
            
            # Likelihood
            #z_exp = tt.exp(z1)
    
            #sof = pm.Deterministic('sof', z_exp/ tt.sum(z_exp))
            sof = pm.Deterministic('sof', tt.nnet.softmax(z))
        
    
            # Defino likelihood en funcion de los valores establecidos y priors
            obs_pos = pm.Multinomial('obs_pos', proba, sof , observed=observed_soft)

            #Elijo el modelo 
            step = pm.HamiltonianMC(target_accept=0.20) # Variable continua

            # Obtengo las muestras posteriores
            trace = pm.sample(300, step=step, tune=800, cores=1, chains=1, compute_convergence_checks=False, progressbar=True)


        Soft_posteriori = (trace['sof'].mean(axis=0))
        Soft_posteriori = Variable(torch.FloatTensor(Soft_posteriori), requires_grad = True)
    
    else:
        Soft_posteriori = F.softmax(tensor, dim=1)
    return (Soft_posteriori)

def log_softmax(tensor,a):
    '''
    Funcion Log_Softmax, utiliza el logaritmo de la softmax.
    :Entrada: Valores en 1D y del tipo tensor.
    
    '''    
    return (F.log_softmax(tensor, dim=1))
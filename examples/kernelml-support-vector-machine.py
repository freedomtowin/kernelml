from sklearn import datasets, preprocessing
import kernelml
from numba import jit,njit, prange, types
import numpy as np
import pandas as pd
import numpy

data = datasets.load_iris()
y = data.target
X = data.data
features = data.feature_names


#the first column is the intercept
X = np.column_stack((np.ones(X.shape[0]),X))

ohe = preprocessing.OneHotEncoder()

y = ohe.fit_transform(y.reshape(-1,1)).toarray()

# y : [-1,1]
y = 2*(y-0.5)



@jit('float64(float64[:,:], float64[:,:], float64[:,:], float64)',nopython=True)
def svm_loss(x,y,w,C):
    w = w.copy()
    w = w.reshape((x.shape[1],y.shape[1]))
    out = 1-y*x.dot(w)
    for i in range(out.shape[1]):
        out[:,i] = C*np.maximum(0,out[:,i])
    
    loss = np.sum(out) + np.sum(w[1:]**2)
    return loss

@njit('float64[:](float64[:,:], float64[:,:], float64[:,:], float64)',parallel=True)
def map_losses(X,y,w_list,C):
    N = w_list.shape[1]
    resX = np.zeros(N)
    for i in prange(N):
        loss = svm_loss(X,y,w_list[:,i:i+1].astype(np.float64),C)
        resX[i] = loss
    return resX



runs = 10
zscore = 1.
simulation_factor = 300
volatility = 10
cycles = 20
volume = 10

kml = kernelml.KernelML(
         prior_sampler_fcn=None,
         posterior_sampler_fcn=None,
         intermediate_sampler_fcn=None,
         mini_batch_sampler_fcn=None,
         parameter_transform_fcn=None,
         loss_calculation_fcn=map_losses,
         batch_size=None)

C = np.float64(1.0)
args_list = [C]

kml.optimize(X,y,
                                args=args_list,
                                number_of_parameters=15,
                                number_of_realizations=runs,
                                number_of_random_simulations = simulation_factor,
                                number_of_cycles=cycles,
                                update_volatility = volatility,
                                update_volume=volume,
                                convergence_z_score=zscore,
                                prior_uniform_low=-1,
                                prior_uniform_high=1,
                                print_feedback=True)

def svm(x,w,):
    out = x.dot(w)
    return out

ytrue = data.target

print('target:',ytrue)

w = kml.kmldata.best_weight_vector.reshape(5,3)

ypred = np.argmax(X.dot(w),axis=1)

print('predicted:',ypred)

print('accuracy:',np.sum(ypred==ytrue)/ytrue.shape[0])

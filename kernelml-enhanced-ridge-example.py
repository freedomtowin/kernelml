
import kernelml
from numba import jit,njit, prange, types
# import seaborn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
import numpy
train=pd.read_csv("DATA/kc_house_train_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test=pd.read_csv("DATA/kc_house_test_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})


def sampler_uniform_distribution(kmldata):
    
    random_samples = kmldata.prior_random_samples
    variances = np.var(kmldata.update_history[:,:],axis=1).flatten()
    means = kmldata.best_weight_vector.flatten()
#     means  = kmldata.update_history[:,-1].flatten()
    return np.vstack([np.random.uniform(mu-np.sqrt(sigma*12)/2,mu+np.sqrt(sigma*12)/2,(random_samples)) for sigma,mu in zip(variances,means)])


@jit('float64(float64[:,:], float64[:,:], float64[:,:])',nopython=True)
def ridge_least_sqs_loss(x,y,w):
    alpha,w = w[0][0],w[1:]
    penalty = 0
    value = 1
    if alpha<value:
        penalty = 3*abs(value-alpha)
    if alpha<0:
        alpha=0
    hypothesis = x.dot(w)
    loss = hypothesis-y 
    return np.sum(loss**2)/len(y) + alpha*np.sum(w[1:]**2) + penalty*np.sum(w[1:]**2)

@njit('float64[:](float64[:,:], float64[:,:], float64[:,:])',parallel=True)
def map_losses(X,y,w_list):
    N = w_list.shape[1]
    resX = np.zeros(N)
    for i in prange(N):
        loss = ridge_least_sqs_loss(X,y,w_list[:,i:i+1])
        resX[i] = loss
    return resX


X_train = train[['sqft_living','bedrooms','bathrooms']].values
y_train = train[['price']].values
X_test = test[['sqft_living','bedrooms','bathrooms']].values
y_test = test[['price']].values
SST_train = np.sum((y_train-np.mean(y_train))**2)
SST_test = np.sum((y_test-np.mean(y_test))**2)

X_train = np.column_stack((np.ones(X_train.shape[0]),X_train))
X_test = np.column_stack((np.ones(X_test.shape[0]),X_test))

runs = 5
zscore = .09
simulation_factor = 100
volatility = 10

min_per_change=0.001

cycles = 100
volume = 100

kml = kernelml.KernelML(
         prior_sampler_fcn=None,
         posterior_sampler_fcn=sampler_uniform_distribution,
         intermediate_sampler_fcn=None,
         mini_batch_sampler_fcn=None,
         parameter_transform_fcn=None,
         loss_calculation_fcn=map_losses,
         batch_size=None)


# args_list = [np.array([1,2,3],dtype=np.float64)]
args_list = []

kml.optimize(X_train,y_train,
                                args=args_list,
                                number_of_parameters=5,
                                number_of_realizations=runs,
                                number_of_random_simulations = simulation_factor,
                                number_of_cycles=cycles,
                                update_volatility = volatility,
                                update_volume=volume,
                                convergence_z_score=zscore,
                                prior_uniform_low=1,
                                prior_uniform_high=2,
                                print_feedback=True)


#Get model performance on validation data
w = kml.model.get_best_param()
alpha,w = w[0],w[1:].reshape(-1,1)
print('alpha:',alpha)
print('w:',w)

yp_train = X_train.dot(w)
SSE_train = np.sum((y_train-yp_train)**2)

yp_test = X_test.dot(w)
SSE_test = np.sum((y_test-yp_test)**2)

#Compare to sklearn.Ridge(alpha=1)
model = linear_model.Ridge(alpha=1)
model.fit(X_train,y_train)
print('kernelml validation r-squared:',1-SSE_test/SST_test)
print('scikit-learn validation r-squared:',model.score(X_test,y_test))

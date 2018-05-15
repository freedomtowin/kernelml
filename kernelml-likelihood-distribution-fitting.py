import pandas as pd
import time
import seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import kernelml
from scipy import stats

train=pd.read_csv("data/kc_house_train_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test=pd.read_csv("data/kc_house_test_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})

#sample parameters from three distributions
def prior_sampler_custom(num_param):
        dist1 = np.random.uniform(1,np.mean(X),size=(num_param,5000))
        dist2 = np.random.normal(np.std(X),np.std(X),size=(num_param,5000))
        dist3 = np.random.uniform(0.1,2*np.mean(X),size=(num_param,5000))
        w = np.hstack((dist1,dist2,dist3))
        return w

def liklihood_loss(x,y,w):
    hypothesis = x
    hypothesis[hypothesis<=0.00001] = 0.00001
    hypothesis[hypothesis>=0.99999] = 0.99999
    loss = -1*((1-y).T.dot(np.log(1-hypothesis)) + y.T.dot(np.log(hypothesis)))/len(y)
    return loss.flatten()[0]

def loss_function(x,y,w):
    alpha1,loc1,scale1 = w[0],w[1],w[2]
    rv = scale1*stats.norm(alpha1,loc1).pdf(x)
    loss = liklihood_loss(rv,y,w)
    return loss


vals, indx = np.histogram(train[['price']].values, normed=False,bins=30)
X = np.linspace(np.min(train[['price']].values),np.max(train[['price']].values),len(vals)) + np.diff(indx)
X = X.reshape(-1,1)
vals = vals.flatten()/np.max(vals)
vals = vals.reshape(-1,1)
model = kernelml.kernel_optimizer(X,vals,loss_function,num_param=3)
#change how the initial parameters are sampled
model.change_prior_sampler(prior_sampler_custom)
#change how many posterior samples are created for each parameter
model.default_random_simulation_params(random_sample_num=100)
model.adjust_optimizer(update_magnitude=100,analyze_n_parameters=30)
model.adjust_convergence_z_score(1.9)
model.kernel_optimize_(plot=True)   

params = model.get_best_parameters()
errors = model.get_best_losses()
update_history = model.get_parameter_update_history()
w = params[np.where(errors==np.min(errors))].T

mean1,std1,scale1 = w[0],w[1],w[2]

plt.stem(X, scale1*stats.norm.pdf(X,mean1,std1),'r', lw=5, alpha=0.6, label='normal pdf')
plt.plot(X,vals)
plt.show()

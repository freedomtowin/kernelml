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

#sample parameters from distribution
#the mean of X seems like a reasonable center for the distribution params
def prior_sampler_custom(weights,num_param,random_sample):
    w = np.random.uniform(np.mean(X),1,size=(num_param,random_sample))
    return w 

def liklihood_loss(x,y,w):
    hypothesis = x
    hypothesis[hypothesis<=0.00001] = 0.00001
    hypothesis[hypothesis>=0.99999] = 0.99999
    loss = -1*((1-y).T.dot(np.log(1-hypothesis)) + y.T.dot(np.log(hypothesis)))/len(y)
    return loss.flatten()[0]

def distribution_loss(x,y,w):
    alpha1,loc1,scale1 = w[0],w[1],w[2]
    rv = scale1*stats.norm(alpha1,loc1).pdf(x)
    loss = liklihood_loss(rv,y,w)
    return loss


y, indx = np.histogram(train[['price']].values, normed=False,bins=30)
X = np.linspace(np.min(train[['price']].values),
                np.max(train[['price']].values),len(y)) + np.diff(indx)
X = X.reshape(-1,1)

y = y.flatten()/np.max(y)
y = y.reshape(-1,1)


runs = 2
tinterations = 10
nupdates = 5

simulation_factor = 100
mutation_factor = 1
breed_factor = 0


kml = KernelML(
         prior_sampler_fcn=prior_sampler_custom,
         sampler_fcn=None,
         intermediate_sampler_fcn=None,
         mini_batch_sampler_fcn=None,
         parameter_transform_fcn=None,
         batch_size=None)

parameter_by_run,loss_by_run = kml.optimize(X,y,loss_function=distribution_loss,
                                num_param=3,
                                args=[],
                                runs=runs,
                                total_iterations=tinterations,
                                n_parameter_updates=nupdates,
                                simulation_factor = simulation_factor,
                                mutation_factor = mutation_factor,
                                breed_factor = breed_factor,
                                prior_uniform_low=1,
                                prior_uniform_high=2,
                                plot_feedback=False,
                                print_feedback=True)


w = parameter_by_run[-1]
mean1,std1,scale1 = w[0],w[1],w[2]
plt.stem(X, scale1*stats.norm.pdf(X,mean1,std1),'r', lw=5, alpha=0.6, label='normal pdf')
plt.plot(X,y)
plt.show()


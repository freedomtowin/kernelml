import pandas as pd
import time
import seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

train=pd.read_csv("data/kc_house_train_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test=pd.read_csv("data/kc_house_test_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})

full = pd.concat([train[['price','date']],test[['price','date']]])
full.sort_values(by='date',inplace=True)
full=full.groupby('date').count()

plt.plot(full[['price']].values)
plt.title("average housing prices by date - full data")
plt.show()

ts = full[:int(len(full)*0.7)].copy()
ts['i'] = np.arange(0,len(ts))
plt.plot(ts[['price']].values)
plt.title("average housing prices by date - train data")
plt.show()

ts_test = full[int(len(full)*0.7):].copy()
ts_test['i'] = np.arange(len(ts),len(ts)+len(ts_test))
plt.plot(ts_test[['price']].values)
plt.title("average housing prices by date - valid data")
plt.show()

def sin_least_sqs_loss(x,y,w):
    hypothesis = w[0]*x[:,0:1] + np.cos(x[:,1:2]*w[1]-w[2])*w[3]
    loss = hypothesis-y
    return np.sum(loss**2)/len(y)


X = ts[['i']].values
y = ts[["price"]].values
model = kernelml.kernel_optimizer(X,y,sin_least_sqs_loss,num_param=4)
model.add_intercept()
#inital random sample with default sampler
model.prior_uniform_random_simulation_params(low=-1,high=1)
#monte carlo simulation parameters
model.default_random_simulation_params(random_sample_num=1000)
#optimizer parameters
model.adjust_optimizer(analyze_n_parameters=10)
model.kernel_optimize_(plot=True)   

import pandas as pd
import time
import seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import kernelml

train=pd.read_csv("DATA/kc_house_train_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test=pd.read_csv("DATA/kc_house_test_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})

def poly_function(x,w):
    hypothesis = w[0]*x[:,0:1] + w[1]*(x[:,1:2]) + w[2]*(x[:,1:2])**w[3]
    return hypothesis
    
def poly_least_sqs_loss(x,y,w):
    hypothesis = poly_function(x,w)
    loss = hypothesis-y 
    return np.sum(loss**2)/len(y)

start_time = time.time()
X_train = train[['sqft_living']].values
y_train = train[["price"]].values
model = kernelml.kernel_optimizer(X_train,y_train,poly_least_sqs_loss,num_param=4)
model.add_intercept()
model.prior_uniform_random_simulation_params(0,2)
model.kernel_optimize_()    
end_time = time.time()
print("time:",end_time-start_time)

#Get the model parameters by iteration
params = model.best_parameters
errors = model.best_losses
params = np.array(params)
y_train = train[["price"]].values
SST_train = np.sum((y_train-np.mean(y_train))**2)/len(y_train)
1-min(errors)/SST_train

#Create train and test datasets
X_train = train[['sqft_living']].values
y_train = train[["price"]].values
X_test = test[['sqft_living']].values
y_test = test[["price"]].values

#Add the intercept
X_train = np.column_stack((np.ones(X_train.shape[0]),X_train))
X_test = np.column_stack((np.ones(X_test.shape[0]),X_test))

#Get the model parameters by iteration
params = model.best_parameters
errors = model.best_losses
params = np.array(params)

#SST for train and test
SST_train = np.sum((y_train-np.mean(y_train))**2)/len(y_train)
SST_test = np.sum((y_test-np.mean(y_test))**2)/len(y_test)

#Create predict outputs
best_w = params[np.where(errors == np.min(errors))].flatten()
train_predicted_output = poly_function(X_train,best_w)
test_predicted_output = poly_function(X_test,best_w)

SSE_train = np.sum((y_train-train_predicted_output)**2)/len(y_train)
SSE_test = np.sum((y_test-test_predicted_output)**2)/len(y_test)

print('train rsquared:',1-SSE_train/SST_train)
print('validation rsquared:',1-SSE_test/SST_test)

import pandas as pd
import time
import seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import kernelml

train=pd.read_csv("data/kc_house_train_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test=pd.read_csv("data/kc_house_test_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})

def poly_function(x,w):
    hypothesis = w[0]*x[:,0:1] + w[1]*(x[:,1:2]) + w[2]*(x[:,1:2])**w[3]
    return hypothesis
    
def poly_least_sqs_loss(x,y,w):
    np=numpy
    def poly_function(x,w):
        hypothesis = w[0]*x[:,0:1] + w[1]*(x[:,1:2]) + w[2]*(x[:,1:2])**w[3]
        return hypothesis
    hypothesis = poly_function(x,w)
    loss = hypothesis-y 
    return np.sum(loss**2)/len(y)


#Create train and test datasets
X_train = train[['sqft_living']].values
y_train = train[["price"]].values

X_test = test[['sqft_living']].values
y_test = test[["price"]].values

#Add the intercept
X_train = np.column_stack((np.ones(X_train.shape[0]),X_train))
X_test = np.column_stack((np.ones(X_test.shape[0]),X_test))


runs = 3
zscore = 2.0
umagnitude = 1
analyzenparam = 5
nupdates = 3
npriorsamples=100
nrandomsamples = 100
tinterations = 10
sequpdate = False



kml.use_ipyparallel(dview)

kml = KernelML(
         prior_sampler_fcn=None,
         sampler_fcn=None,
         intermediate_sampler_fcn=None,
         mini_batch_sampler_fcn=None,
         parameter_transform_fcn=None,
         batch_size=None)


simulation_factor = 100
mutation_factor = 10
breed_factor= 10

parameter_by_run = kml.optimize(X_train,y_train,loss_function=poly_least_sqs_loss,
                                num_param=4,
                                args=[],
                                runs=runs,n_parameter_updates=5,
                                total_iterations=tinterations,
                                simulation_factor = simulation_factor,
                                mutation_factor = mutation_factor,
                                breed_factor = breed_factor,
                                convergence_z_score=zscore,
                                prior_uniform_low=0,
                                prior_uniform_high=2,
                                plot_feedback=False,
                                print_feedback=True)



#Get the model parameters by iteration
params = kml.model.get_param_by_iter()
errors = kml.model.get_loss_by_iter()
update_history = kml.model.get_parameter_update_history()

#SST for train and test
SST_train = np.sum((y_train-np.mean(y_train))**2)/len(y_train)
SST_test = np.sum((y_test-np.mean(y_test))**2)/len(y_test)

#Create predict outputs
best_w = params[np.where(errors == np.min(errors))].T
train_predicted_output = poly_function(X_train,best_w)
test_predicted_output = poly_function(X_test,best_w)

SSE_train = np.sum((y_train-train_predicted_output)**2)/len(y_train)
SSE_test = np.sum((y_test-test_predicted_output)**2)/len(y_test)

print('train rsquared:',1-SSE_train/SST_train)
print('validation rsquared:',1-SSE_test/SST_train)

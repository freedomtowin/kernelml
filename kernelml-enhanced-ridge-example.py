import kernelml
import seaborn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

train=pd.read_csv("data/kc_house_train_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test=pd.read_csv("data/kc_house_test_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})

def ridge_least_sqs_loss(x,y,w):
    alpha,w = w[-1][0],w[:-1]
    penalty = 0
    value = 1
    if alpha<value:
        penalty = 3*abs(value-alpha)
    if alpha<0:
        alpha=0
    hypothesis = x.dot(w)
    loss = hypothesis-y 
    return np.sum(loss**2)/len(y) + alpha*np.sum(w[1:]**2) + penalty*np.sum(w[1:]**2)


X_train = train[['sqft_living','bedrooms','bathrooms']].values
y_train = train[['price']].values
X_test = test[['sqft_living','bedrooms','bathrooms']].values
y_test = test[['price']].values
SST_train = np.sum((y_train-np.mean(y_train))**2)
SST_test = np.sum((y_test-np.mean(y_test))**2)

X_train = np.column_stack((np.ones(X_train.shape[0]),X_train))
X_test = np.column_stack((np.ones(X_test.shape[0]),X_test))

runs = 2
zscore = 2.0
umagnitude = 0.01
analyzenparam = 100
nupdates = 50
npriorsamples=40
nrandomsamples = 600
tinterations = 10
sequpdate = False


kml = kernelml.KernelML(
         prior_sampler_fcn=None,
         sampler_fcn=None,
         intermediate_sampler_fcn=None,
         parameter_transform_fcn=None,
         batch_size=None)

parameter_by_run = kml.optimize(X_train,y_train,loss_function=ridge_least_sqs_loss,
                                num_param=5,
                                args=[],
                                runs=runs,
                                total_iterations=tinterations,
                                analyze_n_parameters=analyzenparam,
                                n_parameter_updates=nupdates,
                                update_magnitude=umagnitude,
                                sequential_update=sequpdate,
                                percent_of_params_updated=1,
                                init_random_sample_num=npriorsamples,
                                random_sample_num=nrandomsamples,
                                convergence_z_score=1,
                                prior_uniform_low=1,
                                prior_uniform_high=2,
                                plot_feedback=False,
                                print_feedback=False)

#Get model performance on validation data
params = kml.model.get_param_by_iter()
errors = kml.model.get_loss_by_iter()
update_history = kml.model.get_parameter_update_history()
w = params[np.where(errors==np.min(errors))].T
alpha,w = w[-1][0],w[:-1]
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

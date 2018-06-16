import pandas as pd
import time
import seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import kernelml

train=pd.read_csv("data/kc_house_train_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test=pd.read_csv("data/kc_house_test_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})


full = pd.concat([train[['price','date']],test[['price','date']]])
full.sort_values(by='date',inplace=True)
full=full.groupby('date').count()

plt.plot(full[['price']].values)
plt.title("average housing prices by date - full data")
plt.show()

ts_train = full[:int(len(full)*0.7)].copy()
ts_train['i'] = np.arange(0,len(ts_train))
plt.plot(ts_train[['price']].values)
plt.title("average housing prices by date - train data")
plt.show()

ts_test = full[int(len(full)*0.7):].copy()
ts_test['i'] = np.arange(len(ts_train),len(ts_train)+len(ts_test))
plt.plot(ts_test[['price']].values)
plt.title("average housing prices by date - valid data")
plt.show()

def sin_non_linear_model(x,w):
    return w[0]*x[:,0:1] + np.cos(x[:,1:2]*w[1]-w[2])*w[3]

def sin_least_sqs_loss(x,y,w):
    np=numpy
    def sin_non_linear_model(x,w):
        return w[0]*x[:,0:1] + np.cos(x[:,1:2]*w[1]-w[2])*w[3]
    hypothesis = sin_non_linear_model(x,w)
    loss = hypothesis-y
    return np.mean(np.abs(loss))

runs = 1
zscore = 2.0
#make the update magnitude large for more variance and quicker convergence
umagnitude = 100
analyzenparam = 100
nupdates = 10
npriorsamples=1000
nrandomsamples = 1000
tinterations = 100
sequpdate = True


kml = kernelml.KernelML(
         prior_sampler_fcn=None,
         sampler_fcn=None,
         intermediate_sampler_fcn=None,
         mini_batch_sampler_fcn=None,
         parameter_transform_fcn=None,
         batch_size=None)

kml.use_ipyparallel(dview)

X_train = ts_train[['i']].values
y_train = ts_train[["price"]].values

X_train = np.column_stack((np.ones(X_train.shape[0]),X_train))

parameter_by_run = kml.optimize(X_train,y_train,loss_function=sin_least_sqs_loss,
                                num_param=4,
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
                                convergence_z_score=zscore,
                                prior_uniform_low=-1,
                                prior_uniform_high=1,
                                plot_feedback=False,
                                print_feedback=False)


### Ensemble Model

#Create train and test datasets
X_train = ts_train[['i']].values
y_train = ts_train[["price"]].values
X_train = np.column_stack((np.ones(X_train.shape[0]),X_train))

X_test = ts_test[['i']].values
y_test = ts_test[['price']].values
X_test = np.column_stack((np.ones(X_test.shape[0]),X_test))

#Get the model parameters by iteration
params = kml.model.get_param_by_iter()
errors = kml.model.get_loss_by_iter()

#Create ensemble of features
feature_num = 5
best_w_arr = errors.argsort()[:feature_num]
predicted_output_as_feature_train = np.zeros((X_train.shape[0],feature_num))
predicted_output_as_feature_test = np.zeros((X_test.shape[0],feature_num))

#Features from last three parameter updates
i=0
for w in params[best_w_arr,:]:
    predicted_output_as_feature_train[:,i] = sin_non_linear_model(X_train,w).flatten()
    predicted_output_as_feature_test[:,i] = sin_non_linear_model(X_test,w).flatten()
    i+=1

linreg = linear_model.Ridge()
linreg.fit(predicted_output_as_feature_train,y_train)
print('train score:',linreg.score(predicted_output_as_feature_train,y_train))

plt.plot(linreg.predict(predicted_output_as_feature_train))
plt.plot(y_train)
plt.title("average housing prices by date - train data")
plt.show()

plt.plot(linreg.predict(predicted_output_as_feature_test))
plt.plot(y_test)
plt.title("average housing prices by date - valid data")
plt.show()

print('validation rsquared:',linreg.score(predicted_output_as_feature_test,y_test))

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
    return w[0]*x[:,0:1] + np.cos(x[:,1:2]*w[1])*w[2] + np.sin(x[:,1:2]*w[1])*w[3]

def sin_mean_loss(x,y,w):
    hypothesis = sin_non_linear_model(x,w)
    loss = hypothesis-y
    return np.mean(np.abs(loss))

#set the window to include a random set of 80% of the data (batch_size=200)
def mini_batch_random_window(X,y,batch_size):
    W = batch_size//2
    center = np.random.randint(W,X.shape[0]-W)
    X_batch = X[center-W:center+W]
    y_batch = y[center-W:center+W]
    return X_batch,y_batch

runs = 10
zscore = 0.5

tinterations = 10
nupdates = 5
sequpdate = True


kml = KernelML(
         prior_sampler_fcn=None,
         sampler_fcn=None,
         intermediate_sampler_fcn=None,
         mini_batch_sampler_fcn=mini_batch_random_window,
         parameter_transform_fcn=None,
         batch_size=int(X_train.shape[0]*0.8))


simulation_factor = 1000
mutation_factor = 1
breed_factor = 3

X_train = ts_train[['i']].values
y_train = ts_train[["price"]].values

X_train = np.column_stack((np.ones(X_train.shape[0]),X_train))

parameter_by_run,loss_by_run = kml.optimize(X_train,y_train,loss_function=sin_mean_loss,
                                num_param=4,
                                args=[],
                                runs=runs,
                                total_iterations=tinterations,
                                n_parameter_updates=nupdates,
                                simulation_factor=simulation_factor,
                                mutation_factor=mutation_factor,
                                breed_factor=breed_factor,
                                convergence_z_score=zscore,
                                prior_uniform_low=-0.001,
                                prior_uniform_high=0.001,
                                plot_feedback=False,
                                print_feedback=True)



plt.plot(parameter_by_run)
plt.show()

plt.plot(loss_by_run)
plt.show()

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

def get_rsq(y,yp):
    return 1-np.sum((yp-y)**2)/np.sum((np.mean(y)-y)**2)

#Create ensemble of features
feature_num = 10
best_w_arr = errors.argsort()[:feature_num]

w = np.mean(parameter_by_run[-10:],axis=0)

plt.plot(sin_non_linear_model(X_test,w).flatten())
plt.plot(y_test)
plt.show()

print(get_rsq(y_test.flatten(),sin_non_linear_model(X_test,w).flatten()))

predicted_output_as_feature_train = np.zeros((X_train.shape[0],feature_num))
predicted_output_as_feature_test = np.zeros((X_test.shape[0],feature_num))

#Features from last three parameter updates
i=0
for w in params[best_w_arr,:]:
    predicted_output_as_feature_train[:,i] = sin_non_linear_model(X_train,w).flatten()
    predicted_output_as_feature_test[:,i] = sin_non_linear_model(X_test,w).flatten()
    i+=1

plt.plot(np.mean(predicted_output_as_feature_test,axis=1))
plt.plot(y_test)
plt.show()


train=pd.read_csv("data/kc_house_train_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test=pd.read_csv("data/kc_house_test_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})

def ridge_least_sqs_loss(x,y,w):
    alpha,w = w[-1][0],w[:-1]
    penalty = 0
    value = 1
    if alpha<=value:
        penalty = 10*abs(value-alpha)
    hypothesis = x.dot(w)
    loss = hypothesis-y 
    return np.sum(loss**2)/len(y) + alpha*np.sum(w[1:]**2) + penalty*np.sum(w[1:]**2)

X_train = train[['sqft_living','bedrooms','bathrooms']].values
y_train = train[['price']].values
X_test = test[['sqft_living','bedrooms','bathrooms']].values
y_test = test[['price']].values
SST_train = np.sum((y_train-np.mean(y_train))**2)
SST_test = np.sum((y_test-np.mean(y_test))**2)

start_time = time.time()
model = kernelml.kernel_optimizer(X_train,y_train,ridge_least_sqs_loss,num_param=5)
model.add_intercept()

model.default_random_simulation_params(prior_uniform_low=1,prior_uniform_high=2)
model.kernel_optimize_()
end_time = time.time()
print("time:",end_time-start_time)

#Get model performance on validation data
params = model.get_best_parameters()
errors = model.get_best_losses()
update_history = model.get_parameter_update_history()
w = params[np.where(errors==np.min(errors))].T
alpha,w = w[-1][0],w[:-1]
print('alpha:',alpha)
print('w:',w)

X = np.column_stack((np.ones(X_train.shape[0]),X_train))
yp_train = X.dot(w)
SSE_train = np.sum((y_train-yp_train)**2)

X = np.column_stack((np.ones(X_test.shape[0]),X_test))
yp_test = X.dot(w)
SSE_test = np.sum((y_test-yp_test)**2)

#Compare to sklearn.Ridge(alpha=1)
X_train = train[['sqft_living','bedrooms','bathrooms']].values
y_train = train[['price']].values
X_test = test[['sqft_living','bedrooms','bathrooms']].values
y_test = test[['price']].values
model = linear_model.Ridge(alpha=1)
model.fit(X_train,y_train)
model.score(X_test,y_test),model.intercept_,model.coef_
print(1-SSE_test/SST_test)

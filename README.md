# Kernel Machine Learning 

## Table of contents
1. [Introduction](#installation)
2. [Examples](#examples)
    1. [Kernel Mapping](#kernelmapping)
    2. [Find Optimal Power Transformation](#powertransformation)
    3. [Find Sinusoidal Parameters](#sinusoids)
    4. [Custom Log Likelihood Loss](#loglikelihood)
3. [Methods](#methods)

## Installation <a name="installation"></a>

```
pip install kernelml
```

## Examples <a name="examples"></a>

### Kernel Mapping <a name="kernelmapping"></a>
Find a projection for latitude and longitude so that the Haversian distance to the centroid of the data points is equal to that of the projected latitude and longitude in Euclidean space.

![](https://user-images.githubusercontent.com/21232362/39224068-37ba94c0-4813-11e8-9414-6d489fe86b4d.png)


```python
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = [x*np.pi/180 for x in [lon1, lat1, lon2, lat2]] 

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def euclid_dist_to_centroid(x,y,w):
        lon1, lat1, lon2, lat2 = x[:,0:1],x[:,1:2],x[:,2:3],x[:,3:4]
        euclid = np.sqrt((w[0]*lon1-w[0]*lon2)**2+(w[1]*lat1-w[1]*lat2)**2) 
        haver = y
        return np.sum((euclid-haver)**2)


train['mean_long'] = np.mean(train[['long']].values)
train['mean_lat'] = np.mean(train[['lat']].values)
train['haversine'] = train[['long','lat','mean_long','mean_lat']].apply(lambda x: haversine(x[0],x[1],x[2],x[3]),axis=1)

start_time = time.time()
X = train[['long','lat','mean_long','mean_lat']].values
y = train[["haversine"]].values
model = kernelml.kernel_optimizer(X,y,euclid_dist_to_centroid,num_param=2)
model.kernel_optimize_(optimizer=kernelml.pid_linear_combination)    
end_time = time.time()
print("time:",end_time-start_time)
```
##### Access Model Parameters and Losses

```python
params = model.best_parameters
error = model.best_losses
params = np.array(params)

SSE = np.min(error)
SST = np.sum((y-np.mean(y))**2)

#plot eculid distance vs haversine
w = params[np.where(error==np.min(error))].flatten()
lon1, lat1, lon2, lat2 = X[:,0:1],X[:,1:2],X[:,2:3],X[:,3:4]
plt.plot(np.sqrt((w[0]*lon1-w[0]*lon2)**2+(w[1]*lat1-w[1]*lat2)**2) ,y,'.')
plt.show()

```

### Non Linear Coefficients - Power Transformation <a name="powertransformation"></a>

```python
def poly_least_sqs_loss(x,y,w):
    hypothesis = w[0]*x[:,0:1] + w[1]*(x[:,1:2]) + w[2]*(x[:,1:2])**w[3]
    loss = hypothesis-y 
    return np.sum(loss**2)/len(y)

start_time = time.time()
X = train[['sqft_living']].values
y = train[["price"]].values
model = kernelml.kernel_optimizer(X,y,poly_least_sqs_loss,num_param=4)
model.add_intercept()
model.adjust_uniform_random_low_high(0,2)
model.kernel_optimize_()    
end_time = time.time()
print("time:",end_time-start_time)
```

### Non Linear Coefficients - Sinusoids <a name="sinusoids"></a>

The optimizer returns a history of parameters for every iteration. Each parameter in the history fits the data slightly differently. Using a combination of the predicted values from these parameters, ensembled together, can improve results.

```python
def sin_least_sqs_loss(x,y,w):
    hypothesis = w[0]*x[:,0:1] + np.cos(x[:,1:2]*w[1]-w[2])*w[3]
    loss = hypothesis-y
    return np.sum(loss**2)/len(y)
```

![](https://user-images.githubusercontent.com/21232362/39224841-34a459fc-4817-11e8-9786-be1c8e2ef595.png)
![](https://user-images.githubusercontent.com/21232362/39224840-323fef32-4817-11e8-9af2-c417b5c78a19.png)



### Custom Loss Function - Loglikelihood <a name="loglikelihood"></a>

```python
def liklihood_loss(x,y,w):
    hypothesis = x.dot(w)
    hypothesis = 1/(1+np.exp(-1*hypothesis))
    hypothesis[hypothesis<=0.00001] = 0.00001
    loss = -1*((1-y).T.dot(np.log(1-hypothesis)) + y.T.dot(np.log(hypothesis)))/len(y)
    return loss.flatten()[0]

X = train[['bedrooms','bathrooms']].values
start_time = time.time()
model = kernelml.kernel_optimizer(X,y,liklihood_loss,num_param=3)
model.add_intercept()
model.adjust_random_simulation(random_sample_num=100)
model.adjust_optimizer(n_parameter_updates=100,analyze_n_parameters=100)
model.kernel_optimize_()
end_time = time.time()
print("time:",end_time-start_time)
```

##### Compare Parameters and Losses with scikit-learn.linear_model.LogisticRegression

```python
X = train[['bedrooms','bathrooms']].values
y = (train['sqft_living'] > np.mean(train['sqft_living'])).reshape(len(train),1)
model = linear_model.LogisticRegression()
model.fit(X, y)
```

## Methods <a name="methods"></a>

```
# Initializes the optimizer
kernelml.kernel_optimizer(X,y,loss_function,num_param)
```
**X:** input matrix

**y:** output vector

**loss_function:** f(x,y,w), outputs loss

**num_param:** number of parameters in the loss function

```
# Begins the optimization process
kernelml.kernel_optimize_(plot=False,print_feedback=True)
```
**plot:** provides real-time plots of parameters and losses

**print_feedback:** real-time feedback of parameters,losses, and convergence


```
# Adjusts the maximum number of iterations
kernelml.adjust_maximum_iterations(self,total_iterations=100) 
```
**total_iterations:** number of iterations (+bias)

```
# Adjusts the initial parameter sampling (this can be useful to avoid underflows or overflows)
kernelml.adjust_uniform_random_low_high(self,low=-1,high=1)
```

```
# Adjusts random simulation of parameters
kernelml.adjust_random_simulation(self,init_random_sample_num=1000, random_sample_num=100)
```
**init_random_sample_num:** the number of initial simulated parameters (+bias)

**random_sample_num:** the number of intermediate simulated parameters (+bias)
   

```
# Adjusts how the optimizer analyzes and updates the parameters
kernelml.adjust_optimizer(self, analyze_n_parameters=20, n_parameter_updates=100, update_magnitude=100)
```
**analyze_n_parameters:** the number of parameters analyzed (+variance)

**n_parameter_updates:** the number of parameter updates per iteration (+bias)

**update_magnitude:** the magnitude of the updates (+variance)

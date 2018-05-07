# Kernel Machine Learning 

## Table of contents
1. [Installation](#installation)
2. [Examples](#examples)
    1. [Kernel Mapping](#kernelmapping)
    2. [Access Model Parameters/Losses](#accessmodel)
    2. [Fit Optimal Power Transformation](#powertransformation)
    3. [Fit Sinusoidal Parameters - Ensemble Model](#sinusoids)
    4. [Custom Log Likelihood Loss](#loglikelihood)
3. [Methods](#methods)
    1. [Adjust Default Random Sampling Parameters](#adjustrandom)
    2. [Override Random Sampling Functions](#simulationdefaults)

## Installation <a name="installation"></a>

```
pip install kernelml
```

## Examples <a name="examples"></a>

### Kernel Mapping <a name="kernelmapping"></a>

Lets take the problem of clustering longitude and latitude coordinates. Clustering methods such as K-means use Euclidean distances to compare observations. However, The Euclidean distances between the longitude and latitude data points do not map directly to Haversine distance. That means if you normalize the coordinate between 0 and 1, the distance won't be accurately represented in the clustering model. A possible solution is to find a projection for latitude and longitude so that the Haversian distance to the centroid of the data points is equal to that of the projected latitude and longitude in Euclidean space.

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
model.kernel_optimize_()    
end_time = time.time()
print("time:",end_time-start_time)
```
### Access Model Parameters and Losses <a name="accessmodel"></a>

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

Another, simpler problem is to find the optimal values of non-linear coefficients, i.e, power transformations in a least squares linear model. The reason for doing this is simple: integer power transformations rarely capture the best fitting transformation. By allowing the power transformation to be any real number, the accuracy will improve and the model will generalize to the validation data better.  

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
model.prior_uniform_random_simulation_params(0,2)
model.kernel_optimize_()    
end_time = time.time()
print("time:",end_time-start_time)
```

### Non Linear Coefficients - Ensemble Model - Sinusoids <a name="sinusoids"></a>

The optimizer returns a history of parameters for every iteration. Each parameter in the history fits the data slightly differently. Using a combination of the predicted values from these parameters, ensembled together, can improve results. In this example, the phase, time shift, and the scaling for the cosine term will update in that order.

```python
def sin_least_sqs_loss(x,y,w):
    hypothesis = w[0]*x[:,0:1] + np.cos(x[:,1:2]*w[1]-w[2])*w[3]
    loss = hypothesis-y
    return np.sum(loss**2)/len(y)
```

The predicted output from each parameter was used as a feature in a unifying model. The train and validation plots below show how using multiple parameter sets can fit complex shapes.

![](https://user-images.githubusercontent.com/21232362/39224841-34a459fc-4817-11e8-9786-be1c8e2ef595.png)
![](https://user-images.githubusercontent.com/21232362/39224840-323fef32-4817-11e8-9af2-c417b5c78a19.png)

The video below shows how the parameters were adjusted for the example above.  Please see the example code in kernelml-time-series-example.py.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/6VJ4KeqJiB4/0.jpg)](https://www.youtube.com/watch?v=6VJ4KeqJiB4)

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
model.default_random_simulation_params(random_sample_num=100)
model.adjust_optimizer(n_parameter_updates=100,analyze_n_parameters=100)
model.kernel_optimize_()
end_time = time.time()
print("time:",end_time-start_time)
```

**Compare Parameters and Losses with scikit-learn.linear_model.LogisticRegression**

```python
X = train[['bedrooms','bathrooms']].values
y = (train['sqft_living'] > np.mean(train['sqft_living'])).reshape(len(train),1)
model = linear_model.LogisticRegression()
model.fit(X, y)
```

## Methods <a name="methods"></a>

```python
# Initializes the optimizer
kernelml.kernel_optimizer(X,y,loss_function,num_param)
```
* **X:** input matrix
* **y:** output vector
* **loss_function:** f(x,y,w), outputs loss
* **num_param:** number of parameters in the loss function

```python
# Begins the optimization process
kernelml.kernel_optimize_(plot=False,print_feedback=True)
```
* **plot:** provides real-time plots of parameters and losses
* **print_feedback:** real-time feedback of parameters,losses, and convergence


```python
# Adjusts the maximum number of iterations
kernelml.adjust_maximum_iterations(self,total_iterations=100) 
```
* **total_iterations:** number of iterations (+bias)

### Adjust Random Sampling Parameters <a name="adjustrandom"></a>

Note: the values in the following functions will be ignored when the random sampling functions are overrided.

```python
# Adjusts the initial parameter sampling (this can be useful to avoid underflows or overflows)
# Not to be confused with kernelml.prior_uniform_random_simulation_distribution which is the actual sampling method
kernelml.prior_uniform_random_simulation_params(self,low=-1,high=1)
```

```python
# Adjusts random simulation of parameters
kernelml.default_random_simulation_params(self,init_random_sample_num=1000, random_sample_num=100)
```
* **init_random_sample_num:** the number of initial simulated parameters (+bias)
* **random_sample_num:** the number of intermediate simulated parameters (+bias)
   

```python
# Adjusts how the optimizer analyzes and updates the parameters
kernelml.adjust_optimizer(self, analyze_n_parameters=20, n_parameter_updates=100, update_magnitude=100)
```
* **analyze_n_parameters:** the number of parameters analyzed (+variance)
* **n_parameter_updates:** the number of parameter updates per iteration (+bias)
* **update_magnitude:** the magnitude of the updates (+variance)

### Override Random Sampling Functions <a name="simulationdefaults"></a>

The default random sampling functions for the prior and posterior distributions can be overrided. User defined random sampling function must have the same parameters as the default. Please see the default random sampling functions below. 

```python
    #inital parameter sampler (default)
    def prior_sampler_uniform_distribution(self,num_param):
        return np.random.uniform(low=self.low,high=self.high,size=(num_param,self.init_random_sample_num))

    #multivariate normal sampler (default)
    def sampler_multivariate_normal_distribution(self,best_param,
                                                param_by_iter,
                                                error_by_iter,
                                                parameter_update_history,
                                                random_sample_num=100):
        covariance = np.diag(np.var(parameter_update_history[:,:],axis=1))
        best = param_by_iter[np.where(error_by_iter==np.min(error_by_iter))[0]]
        mean = best.flatten()
        try:
            return np.random.multivariate_normal(mean, covariance, (random_sample_num)).T
        except:
            print(best,np.where(error_by_iter==np.min(error_by_iter)))
```

```python
    #override functions
    def change_random_sampler(self,fcn):
        self.sampler = fcn
        
    def change_prior_sampler(self,fcn):
        self.prior_sampler = fcn
```            

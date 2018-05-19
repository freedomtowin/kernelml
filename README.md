# Kernel Machine Learning 

Project Status: Beta

Current Version: 2.514

## Table of contents
1. [Installation](#installation)
2. [Examples](#examples)
    1. [Kernel Mapping](#kernelmapping)
    2. [Fit Optimal Power Transformation](#powertransformation)
    3. [Fit Sinusoidal Parameters - Ensemble Model](#sinusoids)
    4. [Negative Log Likelihood Loss](#loglikelihood)
    5. [Enhanced Ridge Regression](#ridge)
3. [Methods](#methods)
    1. [Access Model Parameters/Losses](#accessmodel)
    2. [Convergence](#convergence)
    3. [Parameter Transforms](#transforms)
    3. [Adjust Default Random Sampling Parameters](#adjustrandom)
    4. [Adjust Optimizer Parameters](#adjustopt)
    5. [Override Random Sampling Functions](#simulationdefaults)

## Installation <a name="installation"></a>

```
pip install kernelml
```

## Examples <a name="examples"></a>

### Kernel Mapping <a name="kernelmapping"></a>

Lets take the problem of clustering longitude and latitude coordinates. Clustering methods such as K-means use Euclidean distances to compare observations. However, The Euclidean distances between the longitude and latitude data points do not map directly to Haversine distance. That means if you normalize the coordinate between 0 and 1, the distance won't be accurately represented in the clustering model. A possible solution is to find a projection for latitude and longitude so that the Haversian distance to the centroid of the data points is equal to that of the projected latitude and longitude in Euclidean space. Please see kernelml-haversine-to-euclidean.py.

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
```

### Non Linear Coefficients - Power Transformation <a name="powertransformation"></a>

Another, simpler problem is to find the optimal values of non-linear coefficients, i.e, power transformations in a least squares linear model. The reason for doing this is simple: integer power transformations rarely capture the best fitting transformation. By allowing the power transformation to be any real number, the accuracy will improve and the model will generalize to the validation data better. Please see kernelml-power-transformation-example.py.

```python
def poly_least_sqs_loss(x,y,w):
    hypothesis = w[0]*x[:,0:1] + w[1]*(x[:,1:2]) + w[2]*(x[:,1:2])**w[3]
    loss = hypothesis-y 
    return np.sum(loss**2)/len(y)
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

The video below shows how the parameters were adjusted for the example above.  Please see the example code in kernelml-time-series-example.py. Ensemble model results may vary due to the random nature of parameter updates.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/6VJ4KeqJiB4/0.jpg)](https://www.youtube.com/watch?v=6VJ4KeqJiB4)

### Negative Loglikelihood Loss <a name="loglikelihood"></a>

Kernelml can minimize non-linear loss functions such as minimizing the negative loglikelihood for logistic regression. In addition, the optimizer fit a distribution to an empirical histogram using negative loglikelihood as the loss function. Please see kernelml-likelihood-distribution-fitting.py. 

```python
def logistic_liklihood_loss(x,y,w):
    hypothesis = x.dot(w)
    hypothesis = 1/(1+np.exp(-1*hypothesis))
    hypothesis[hypothesis<=0.00001] = 0.00001
    loss = -1*((1-y).T.dot(np.log(1-hypothesis)) + y.T.dot(np.log(hypothesis)))/len(y)
    return loss.flatten()[0]
```

### Enhanced Ridge Regression <a name="ridge"></a>

Add a parameter for L2 regularization and allow the alpha parameter to fluxuate from a target value. The added flexibility can improve generalization to the validation data. Please see kernelml-enhanced-ridge-example.py.

```python
def ridge_least_sqs_loss(x,y,w):
    alpha,w = w[-1][0],w[:-1]
    penalty = 0
    value = 1
    if alpha<=value:
        penalty = 10*abs(value-alpha)
    hypothesis = x.dot(w)
    loss = hypothesis-y 
    return np.sum(loss**2)/len(y) + alpha*np.sum(w[1:]**2) + penalty*np.sum(w[1:]**2)
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
# Appends an array of ones to the left hand side of the numpy input matrix
model.add_intercept()
```
### Access Model Parameters and Losses <a name="accessmodel"></a>

```python
params = model.get_best_parameters()
errors = model.get_best_losses()
update_history = model.get_parameter_update_history()
best_w = params[np.where(errors==np.min(errors))].flatten()
```

### Convergence <a name="convergence"></a>

The model saves the best parameter and user-defined loss after each iteration. The model also record a history of all parameter updates. The question is how to use this data to define convergence. One possible solution is:

```python
#Convergence algorithm
convergence = (best_parameter-np.mean(param_by_iter[-10:,:],axis=0))/(np.std(param_by_iter[-10:,:],axis=0))

if np.all(np.abs(convergence)<1):
    print('converged')
    break
 ```
The formula creates a Z-score using the last 10 parameters and the best parameter. If the Z-score for all the parameters is less than 1, then the algorithm can be said to have converged. This convergence solution works well when there is a theoretical best parameter set.

```python
model.adjust_convergence_z_score(z=1)
```
* **z:** the z score -  defines when the algorithm converges

### Parameter Transforms <a name="transform"></a>

The default parameter tranform function can be overrided. The parameters can be transformed before the loss calculations and parameter updates. For example, the parameters can be transformed if some parameters must be integers, positive, or within a certain range.

```python
# The default parameter transform return the parameter set unchanged 
model.default_parameter_transform(w):
    return w

# Change the default parameter transform
model.change_parameter_transform(fcn)
```
* **w:** the parameter set used to calculate loss

### Adjust Random Sampling Parameters <a name="adjustrandom"></a>

Note: the values in the following functions will be ignored when the random sampling functions are overrided.

```python
# Adjusts random simulation of parameters
model.default_random_simulation_params(self,init_random_sample_num=1000,
                                            random_sample_num=100,prior_uniform_low=-1,prior_uniform_high=1)
```
* **init_random_sample_num:** the number of initial simulated parameters (+bias)
* **random_sample_num:** the number of intermediate simulated parameters (+bias)
* **prior_uniform_low:** default pior random sampler - uniform distribution - low
* **prior_uniform_high:** default pior random sampler - uniform distribution - high
   
### Adjust Optimizer Parameters  <a name="adjustopt"></a>

```python
# Adjusts how the optimizer analyzes and updates the parameters
model.adjust_optimizer(self,total_iterations=100,
                            analyze_n_parameters=20,
                            n_parameter_updates=100,
                            update_magnitude=100,
                            sequential_update=True)
```
* **total_iterations:** number of iterations (+bias)
* **analyze_n_parameters:** the number of parameters analyzed (+variance)
* **n_parameter_updates:** the number of parameter updates per iteration (+bias)
* **update_magnitude:** the magnitude of the updates - corresponds to magnitude of loss function (+variance)
* **sequential_update:** controls whether the parameters are updated sequentially or randomly

### Override Random Sampling Functions <a name="simulationdefaults"></a>

The default random sampling functions for the prior and posterior distributions can be overrided. User defined random sampling function must have the same parameters as the default. Please see the default random sampling functions below. 

```python
    #inital parameter sampler (default)
    def prior_sampler_uniform_distribution(num_param):
        return np.random.uniform(low=self.low,high=self.high,size=(num_param,self.init_random_sample_num))

    #multivariate normal sampler (default)
    def sampler_multivariate_normal_distribution(best_param,
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

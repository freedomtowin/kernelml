# Kernel Machine Learning 

Project Status: Beta

Current Version: 3.17

Examples script will be updated soon.

## About 

KernelML is brute force optimizer that can be used to train machine learning models. The package uses a combination of a neuroevolution algorithms, heuristics, and monte carlo simulations to optimize a parameter vector with a user-defined loss function.

The field of high performance computing is always change, and parallelization is very important for simulation based optimizers. This is why parallelization is left to the developer to implement. KernelML supports Numba compiled functions for parallelization across CPUs and GPU acceleration. 

Due to the recent changes to threading and MKL in Anaconda distributions additional steps it has become difficult to incoporate the original parallelization engine, ipyparallel. However, it is still possible to use ipyparallel with KernelML with nomkl libraries. 

The following script have not been updated for KernelML 3.1:
1. kernelml-autoencoder-example.ipynb
2. kernelml-time-series-example.py

## Table of contents
1. [Installation](#installation)
2. [Examples](#examples)
    1. [Kernel Mapping](#kernelmapping)
    2. [Optimal Power Transformation](#powertransformation)
    3. [Negative Log Likelihood Loss](#loglikelihood)
    4. [Enhanced Ridge Regression](#ridge)
    5. [Parameter Tuning](#tuning)
3. [Methods](#methods)
    1. [KmlData](#kmldata)
    2. [Passing Static Data](#staticdata)
    2. [Parallel Processing with Numba](#parallelnumba)
    2. [Convergence](#convergence)
    3. [Override Random Sampling Functions](#simulationdefaults)
    4. [Parameter Transforms](#transforms)
4. [Extensions](#ext)
    1. [High Density Region Estimation](#hdr)

## Installation <a name="installation"></a>

```
pip install kernelml --upgrade
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
        
# y: pre-computed haversine distance to centroid
# x: lon1, lat1, lon2, lat2: input longitude, input latitiude, centroid longitude, centroid latitiude
```

### Non Linear Coefficients - Power Transformation <a name="powertransformation"></a>

Another, simpler problem is to find the optimal values of non-linear coefficients, i.e, power transformations in a least squares linear model. The reason for doing this is simple: integer power transformations rarely capture the best fitting transformation. By allowing the power transformation to be any real number, the accuracy will improve and the model will generalize to the validation data better. Please see kernelml-power-transformation-example.py.

```python
def poly_least_sqs_loss(x,y,w):
    hypothesis = w[0]*x[:,0:1] + w[1]*(x[:,1:2]) + w[2]*(x[:,1:2])**w[3]
    loss = hypothesis-y 
    return np.sum(loss**2)/len(y)
```

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
    alpha,w = w[0][0],w[1:]
    penalty = 0
    value = 1
    if alpha<value:
        penalty = 3*abs(value-alpha)
    if alpha<0:
        alpha=0
    hypothesis = x.dot(w)
    loss = hypothesis-y 
    return np.sum(loss**2)/len(y) + alpha*np.sum(w[1:]**2) + penalty*np.sum(w[1:]**2)
```

### Parameter Tuning "Rules of Thumb" <a name="tuning"></a>

There are many potential strategies for choosing optimization parameters. However, the choice of different setting involves balancing the number_of_random_simulations, update_volume, and update_volatility.



## Methods <a name="methods"></a>


### KernelML <a name="intialization"></a>

```python
kml = kernelml.KernelML(
                 loss_calculation_fcn,
                 prior_sampler_fcn=None,
                 posterior_sampler_fcn=None,
                 intermediate_sampler_fcn=None,
                 parameter_transform_fcn=None,
                 batch_size=None)
```

* **loss_calculation_fcn:** this function defines how the loss function is calculated 
* **prior_sampler_fcn:** this function defines how the initial parameter set is sampled 
* **posterior_sampler_fcn:** this function defines how the parameters are sampled between interations
* **intermediate_sampler_fcn:** this function defines how the priors are set between runs
* **parameter_transform_fcn:** this function transforms the parameter vector before processing
* **batch_size:** defines the random sample size before each run (default is all samples)

`loss_calculation_fcn=map_losses` should be structured as follows:

```python
def least_sq_loss_function(X,y,w):
   intercept,w = w[0],w[1:]
   hyp = X.dot(w)+intercept
   error = (hyp-y)
   return np.sum(error**2)

def map_losses(X,y,w_list):
    N = w_list.shape[1]
    out = np.zeros(N)
    for i in range(N):
        loss = least_sq_loss_function(X,y,w_list[:,i:i+1])
        out[i] = loss
    return out
```

The `loss_calculation_fcn` should map a list of input parameters to a list of loss function outputs. The `map_losses` function above calls the `least_sq_loss_function` and then stores the output for each parameter set in w_list.

```python
# Begins the optimization process, returns (list of parameters by run),(list of losses by run)
kml.optimize(   X,
                                                y,
                                                number_of_parameters,
                                                args=[],
                                                number_of_realizations=1,
                                                number_of_cycles=20,
                                                update_volume=10,
                                                number_of_random_simulations = 1000,
                                                update_volatility = 1,
                                                convergence_z_score=1,
                                                analyze_n_parameters=None,
                                                update_magnitude=None,
                                                prior_random_samples=None,
                                                posterior_random_samples=None,
                                                prior_uniform_low=-1,
                                                prior_uniform_high=1,
                                                print_feedback=True)
```
* **X:** input matrix
* **y:** output vector
* **loss_function:** f(x,y,w), outputs loss
* **num_param:** number of parameters in the loss function
* **arg:** list of extra data to be passed to the loss function

### Iteration Parameters
* **number_of_realizations:** number of runs
* **number_of_cycles:** number of iterations (+bias)
* **update_volume:** the volume of attempted parameter updates (+bias)

### Learning Rate Parameters
The optimizer's parameters can be automatically adjusted by adjusting the following parameters:
* **number_of_random_simulations:** The number of random simulations per cycle
* **update_volatility:** increases the amount of coefficient augmentation, increases the search magnitude

### Automatically adjusted parameters
* **analyze_n_parameters:** the number of parameters analyzed (+variance)
* **update_magnitude:** the magnitude of the updates - corresponds to magnitude of loss function (+variance)
* **prior_random_sample_num:** the number of prior simulated parameters (+bias)
* **posterior_random_sample_num:** the number of posterior simulated parameters (+bias)

### Optinal Parameters
* **convergence_z_score:** an optional stopping threshold, stops a realization (see Convergence below)
* **min_loss_per_change:** an optional stopping threshold, stops a realization when the change in loss is below threshold
* **prior_uniform_low:** default pior random sampler - uniform distribution - low
* **prior_uniform_high:** default pior random sampler - uniform distribution - high
* **print_feedback:** real-time feedback of parameters,losses, and convergence


### kmldata <a name="kmldata"></a>

A KmlData class is primarily used to pass relevant information to the sampler functions.

### Data
* **KernelML().kmldata.best_weight_vector:** numpy array of the best weights for the current cycle 
* **KernelML().kmldata.current_weights:** numpy array of simulated weights for the current cycle
* **KernelML().kmldata.update_history:** numpy array of parameter updates for the current realization - (weight set, update #)
* **KernelML().kmldata.loss_history:** numpy array of losses for the current realization
* **KernelML().kmldata.realization_number:** current number of realizations
* **KernelML().kmldata.cycle_number:** current number of cycles
* **KernelML().kmldata.number_of_parameters:** number of parameters passed to the loss function
* **KernelML().kmldata.prior_random_samples:** the number of prior simulated parameters
* **KernelML().kmldata.posterior_random_samples:** the number of posterior simulated parameters
* **KernelML().kmldata.number_of_updates:** number of successful parameter updates

### Load kmldata

An optimization model can continue where the previous model left off by loading the kmldata.

```
#optimize
save_kmldata = KernelML().kmldata
#load kmldata
KernelML().load_kmldata(save_kmldata)
```

### Passing Static Data <a name="staticdata"></a>

The args parameter can be set in the kernelml.KernelML().optimize to pass extra data

```python
# For example, if args = [arg1,arg2]

def loss_function(X,y,w,arg1,arg2):
   return loss

def map_losses(X,y,w_list,arg1,arg2):
    N = w_list.shape[1]
    out = np.zeros(N)
    for i in range(N):
        loss = loss_function(X,y,w_list[:,i:i+1],arg1,arg2)
        out[i] = loss
    return out
```

### Parallel Processing with Numba <a name="parallelnumba"></a>

Starting with KernelML 3.0, parallelization can be defined with nuba for GPU or CPU parallelization.

```python

from numba import jit,njit, prange

@jit('float64(float64[:,:], float64[:,:], float64[:,:])',nopython=True)
def least_sq_loss_function(X,y,w):
   intercept,w = w[0],w[1:]
   hyp = X.dot(w)+intercept
   error = (hyp-y)
   return np.sum(error**2)


@njit('float64[:](float64[:,:], float64[:,:], float64[:,:])',parallel=True)
def map_losses(X,y,w_list,alpha):
    N = w_list.shape[1]
    out = np.zeros(N)
    for i in prange(N):
        loss = least_sq_loss_function(X,y,w_list[:,i:i+1])
        out[i] = loss
    return out
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

Another possible solution is to stop the realization when the the % change between cycles is less than a threshold. See `min_loss_per_change`.

### Override Random Sampling Functions <a name="simulationdefaults"></a>

The default random sampling functions for the prior and posterior distributions can be overrided. User defined random sampling function must have the same parameters as the default. Please see the default random sampling functions below. 

```python
    #prior sampler (default)
    def prior_sampler_uniform_distribution(kmldata):
        random_samples = kmldata.prior_random_samples
        num_params = kmldata.number_of_parameters
        # (self.high and self.low) correspond to (prior_uniform_low, prior_uniform_high)
        return np.random.uniform(low=self.low,high=self.high,size=(num_params,
                                                                   random_samples))

    #posterior sampler (default)
    def posterior_sampler_uniform_distribution(kmldata):
    
        random_samples = kmldata.prior_random_samples
        variances = np.var(kmldata.update_history[:,:],axis=1).flatten()
        means = kmldata.best_weight_vector.flatten()

        return np.vstack([np.random.uniform(mu-np.sqrt(sigma*12)/2,mu+np.sqrt(sigma*12)/2,(random_samples)) for sigma,mu in zip(variances,means)])


            
            
    #intermediate sampler (default)
    def intermediate_sampler_uniform_distribution(kmldata):

            random_samples = kmldata.prior_random_samples
            variances = np.var(kmldata.update_history[:,:],axis=1).flatten()
            means = kmldata.update_history[:,-1].flatten()

            return np.vstack([np.random.uniform(mu-np.sqrt(sigma*12)/4,mu+np.sqrt(sigma*12)/4,(random_samples)) for sigma,mu in zip(variances,means)])

        
        
    #mini batch random choice sampler
    def mini_batch_random_choice(X,y,batch_size):
        all_samples = np.arange(0,X.shape[0])
        rand_sample = np.random.choice(all_samples,size=batch_size,replace=False)
        X_batch = X[rand_sample]
        y_batch = y[rand_sample]
        return X_batch,y_batch
```

### Parameter Transforms <a name="transform"></a>

The default parameter tranform function can be overrided. The parameters can be transformed before the loss calculations and parameter updates. For example, the parameters can be transformed if some parameters must be integers, positive, or within a certain range. The argument list in the KernelML().optimize function are passed as parameters.

```python
# The default parameter transform return the parameter set unchanged 
def default_parameter_transform(w,*args):
    # w[rows,columns] = (parameter set,iteration)
    return w
```

## Extensions <a name="ext"></a>

### Critical Region Estimation <a name="hdr"></a>

```python
model = kernelml.region_estimator.HighDensityRegions(number_of_clusters, bins_per_dimension=21, number_of_random_simulations=500, number_of_realizations=10,
                                    smoothing_parameter=2.0)
```                                    
                                    
* **number_of_clusters:** The number of clusters     
* **bins_per_dimension:** The number of histogram bins across each dimensions. This is used for estimating the kernel density function  
* **number_of_realizations:** number of runs                                    
* **number_of_random_simulations:** The number of random simulations per cycle  
* **smoothing_parameter:** Increases the bandwidth of the kernel density estimation 

```python
model.optimize(X,y=None,agg_func='mean',dview=None)
```

This method runs the high region density estimator.

* **X:** Input data -> (rows, columns)
* **y:** target data -> (rows, columns)
* **agg_func:** The aggregate function for the target variable y: 'mean', 'variance', 'max', 'false-positive-cost', 'false-negative-cost', 'count'

```python
assignments = model.get_assignments(X,pad=1.0)
```
Returns an assignment matrix (observations, clusters) that represents whether a data point is within a hypercube cluster.

* **X:** Input data -> (rows, columns)
* **pad:** This pads the variance of each HDR cluster.

```python
distance = model.get_distances(X,distance='chebyshev',pad=1.0)
```

Computes the distances between the data points and the hypercube centroids.

* **X:** Input data -> (rows, columns)
* **distance:** the distance metric used to assign data to clusters: 'chebyshev', 'euclidian','mae'
* **pad:** This pads the variance of each HDR cluster.

# Kernel Machine Learning 

### Installation

```
pip install kernelml
```

### Examples
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

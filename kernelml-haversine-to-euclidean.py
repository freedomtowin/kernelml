
import pandas as pd
import time
import seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import kernelml

train=pd.read_csv("DATA/kc_house_train_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test=pd.read_csv("DATA/kc_house_test_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})

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

runs = 3
titerations = 5
nupdates = 2
simulation_factor = 100
mutation_factor = 1
breed_factor = 0

kml = kernelml.KernelML(
         prior_sampler_fcn=None,
         sampler_fcn=None,
         intermediate_sampler_fcn=None,
         mini_batch_sampler_fcn=None,
         parameter_transform_fcn=None,
         batch_size=None)

parameter_by_run = kml.optimize(X,y,loss_function=euclid_dist_to_centroid,
                                num_param=2,
                                args=[],
                                runs=runs,
                                total_iterations=titerations,
                                n_parameter_updates=nupdates,
                                simulation_factor = simulation_factor,
                                mutation_factor = mutation_factor,
                                breed_factor = breed_factor,
                                prior_uniform_low=-1,
                                prior_uniform_high=1,
                                plot_feedback=False,
                                print_feedback=True)



  
end_time = time.time()
print("time:",end_time-start_time)

SST = np.sum((y-np.mean(y))**2) 
params = kml.model.get_param_by_iter()
errors = kml.model.get_loss_by_iter()
update_history = kml.model.get_parameter_update_history()

#R-squared by iteration
1 - errors/SST

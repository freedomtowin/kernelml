#Run this code block only once
from ipyparallel import Client
rc = Client(profile='default')
dview = rc[:]

dview.block = True

with dview.sync_imports():
    import numpy as np
    from scipy import stats
############################

full = pd.read_csv('DATA/hb_training.csv')
# test = pd.read_csv('DATA/titantic_test.csv')

def change_label(x):
    if x =='s':
        return 1
    else: 
        return 0
    
full['Label'] = full['Label'].apply(change_label)
full.drop(['EventId'],axis=1,inplace=True)
features = list(full.columns[:-2])
target = list(full.columns[-1:])

all_samples=full.index
ones = full[full[target].values==1].index
zeros = full[full[target].values==0].index
ones_rand_sample = np.random.choice(ones, size=int(len(ones)*0.5),replace=False)
zeros_rand_sample = np.random.choice(zeros, size=int(len(zeros)*0.5),replace=False)
rand_sample  = np.concatenate((ones_rand_sample,zeros_rand_sample))

np.random.shuffle(rand_sample)

test_sample = np.setdiff1d(all_samples,rand_sample)
valid = full.loc[test_sample,:]
train = full.loc[rand_sample,:]

from scipy import stats

class NNShapeHelper():

    def __init__(self,layer_shape,num_inputs,num_outputs):
        
        self.N_inputs = num_inputs
        self.N_outputs = num_outputs
        self.layer_shape = layer_shape
        self.N_layers = len(layer_shape)
        self.model_shape = []
        self.parameter_shape = []
        
    def get_N_parameters(self):
        
        self.model_shape.append(self.N_inputs)
        input_n_parameters = self.N_inputs*self.layer_shape[0]
        N =  input_n_parameters
        self.parameter_shape.append(input_n_parameters)
        
        for i in range(1,self.N_layers):
            layer_n_parameters = self.layer_shape[i-1]*self.layer_shape[i]
            self.model_shape.append(self.layer_shape[i])
            self.parameter_shape.append(layer_n_parameters)
            N += layer_n_parameters
            
        output_n_parameters = self.N_outputs*self.layer_shape[-1]
        N += output_n_parameters
        self.model_shape.append(self.N_outputs)
        self.parameter_shape.append(output_n_parameters)
        self.N_parameters = N
        return N
        
n = NNShapeHelper([10,len(features)],len(features),1)
num_parameters = n.get_N_parameters()

def autoencoder_function(X,y,w_tensor,predict=False):
    from scipy import stats
    import numpy as np
    
    # define the loss function between predicted output actual output
    def nn_autoencoder_loss(hypothesis,y):
        return np.sum((hypothesis-y)**2)/y.size

    def reshape_vector(w):
        reshape_w = []
        indx = 0
        #If we are using parallel computations, we need to hard code (model shape, parameter, shape)
        for shape,num in zip([30, 30, 1], [300, 300, 30]):
            x = w[indx:num+indx]
            if x.size!=num:
                continue
            x = x.reshape(shape,int(num/shape))
            reshape_w.append(x)
            indx = indx+num
        extra_w = w[indx:]
        return reshape_w,extra_w
        
    #Specifies the way the tensors are combined with the inputs
    def combine_tensors(X,w_tensor):
        w_tensor,extra_w = reshape_vector(w_tensor)
        b1,a1,b2,a2 = extra_w[:4]
        pred = X.dot(w_tensor[0])
        pred = a1*(pred+b1)
        pred = pred.dot(w_tensor[1].T)
        pred = a2*(pred+b2)
        return pred

    #we cannot modify pickled memory so create a copy of the parameter vector
    w_tensor_copy = w_tensor.copy()
    pred = combine_tensors(X,w_tensor_copy)
    if predict==True:
        return pred
    loss = nn_autoencoder_loss(pred,y)
    return loss
    
    
#helper class that specifies the prior sampling of parameters
class RandomPriorSampler():
    
    def __init__(self,w):
        self.w = w
        
    def nn_uniform_distribution(self,num_param):
        result = []
        for i in range(num_param):
            x = np.random.uniform(self.w[i]-0.1*self.w[i],self.w[i]+0.1*self.w[i],size=(1,10000)).T
            result.append(x)
        result = np.squeeze(np.array(result))
        return result

zscore = 2.0
umagnitude = 0.00001
analyzenparam = 500
nupdates = 1
nrandomsamples = 6000
tinterations = 5
encodings = []
for run in range(10):
    #sample 5% of the training data at a time
    all_samples = train.index
    rand_sample = np.random.choice(all_samples, size=int(len(all_samples)*0.05),replace=False)
    test_sample = np.setdiff1d(all_samples,rand_sample)
    test_batch = train.loc[test_sample,:]
    train_batch = train.loc[rand_sample,:]
    #create the test and train batches
    X = train_batch[features].values
    y = train_batch[target].values
    X_test = test_batch[features].values
    y_test = test_batch[target].values

    start_time = time.time()
    #create the model with (input,output,loss_function,number of parameters)
    #we need 4 extra parameters to shift and scale the output after each filter
    model = kernelml.kernel_optimizer(X,X,autoencoder_function,num_param=num_parameters+4)
    #specify that you want to ipyparallel
    model.init_ipyparallel(dview,use_ipyparallel=True)
    #optimizer parameters
    model.adjust_optimizer(sequential_update=False,update_magnitude=umagnitude,
                           analyze_n_parameters=analyzenparam,
                           n_parameter_updates=nupdates,
                          total_iterations=tinterations)
    #convergence parameter
    model.adjust_convergence_z_score(zscore)
    #random simulation parameters
    model.default_random_simulation_params(prior_uniform_low=-1,prior_uniform_high=1,init_random_sample_num=10000,
                                    random_sample_num=nrandomsamples)

    if run>0:
        #this overrides the default simulation function
        #we will use the best parameter set from the previous run as the priors for the current
        model.change_prior_sampler(sampler.nn_uniform_distribution)

    #start optimizer
    model.kernel_optimize_()    
    end_time = time.time()
    
    end_time = time.time()
    print('time',end_time-start_time)

    #get the best parameter set
    params = model.get_param_by_iter()
    errors = model.get_loss_by_iter()
    best_w_arr = errors.argsort()[0]
    w = params[best_w_arr].T

    #append the results to the prior sampler function & parameter history
    sampler = RandomPriorSampler(w)
    encodings.append(w)
    
X = train[features].values
y = train[target].values
X_test = valid[features].values
y_test = valid[target].values

autoencoder_SST_train = np.sum((X - np.mean(X,axis=0))**2)/X.size
autoencoder_SST_test = np.sum((X_test - np.mean(X,axis=0))**2)/X_test.size
i=0
for w in encodings[:]:
    
    autoencoder_SSE_train = autoencoder_function(X,X,w)
    autoencoder_SSE_test = autoencoder_function(X_test,X_test,w)
    print('iteration',i,'train rsquared',1-autoencoder_SSE_train/autoencoder_SST_train)
    print('iteration',i,'test rsquared',1-autoencoder_SSE_test/autoencoder_SST_test)
    i+=1
    
X_prime = None
X_test_prime = None
for w in encodings[:]:
    
    #just for fun, we are going to use the latent variables in a predictive model
    if X_prime is None:
        X_prime = get_latent_encoding(X,w)
        X_test_prime = get_latent_encoding(X_test,w)
    else:
        X_prime = np.column_stack((X_prime,get_latent_encoding(X,w)))
        X_test_prime = np.column_stack((X_test_prime,get_latent_encoding(X_test,w)))
        
#add the original variables to the features
X_prime = np.column_stack((X,X_prime))
X_test_prime = np.column_stack((X_test,X_test_prime))

from sklearn import neural_network
neural = neural_network.MLPClassifier(hidden_layer_sizes=(100,),early_stopping=False, max_iter=3000)
neural.fit(X_prime,y.ravel())
train_acc = neural.score(X_prime,y.ravel())
test_acc = neural.score(X_test_prime,y_test.ravel())
print('scikit-learn nueral net train score:',train_acc)
print('scikit-learn nueral net test score:',test_acc)
       

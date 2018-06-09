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
    

import time
import numpy as np
import numba
from numba import jit,njit, prange, types
import warnings
from kernelml import KernelML
import sys
import copy
from functools import partial
from .hdr_helpers_bycython import *

class ArgumentError(Exception):
    pass

class VersionError(Exception):
    pass

    
warnings.filterwarnings('ignore')    

### HDRE
class DensityFactorization():

    def __init__(self,number_of_clusters,bins_per_dimension=41,smoothing_parameter=3.0):
        self.num_clusters = number_of_clusters
        self.smoothing_parameter = smoothing_parameter
        self.bins_per_dim = bins_per_dimension
        self.kmldata = None
        self.kde_target = None
        self.norm = 1

        half = (self.bins_per_dim)//2
        mesh = np.meshgrid(*[np.arange(-(half),half+1,1) for _ in range(2)])
        mesh = [d**2 for d in mesh]
        sigma = self.smoothing_parameter
        kernel = np.exp(-sum(mesh)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
        kernel = kernel/np.sum(kernel)
        self.fftkernel = np.fft.fftn(kernel)
        
        
    def load_kde(self,kde):
        self.kde_target = kde
        
    def update_kde(self,X,y=None,alpha=0.5):
        if self.kde_target is not None:
        
            combo_len = len(self.dim_combos)
            
            count=0
            _y_ = None
            for i,j in self.dim_combos:
                _X_ = X[:,[i,j]]
                bins3 = [self.dim_bins[i],self.dim_bins[j]]
                if y is not None and self.multi_dim==True:
                    _y_ = y[:,[i,j]]
                data,_ = hdre_histogram(_X_,y,agg_func='count',cost=None,bins=bins3)
                data=data/np.sum(data)
                new_kde = np.fft.fftshift(np.real(np.fft.ifftn(np.fft.fftn(data)*self.fftkernel)))
                self.kde_target[count]=self.kde_target[count]*(1-alpha) + new_kde*(alpha)
                count+=1

    def optimize(self,X,y=None,cost=None, number_of_random_simulations=1000, number_of_realizations=10, agg_func='count',verbose=True,
        multi_dim=False,
        ):
        
        self.simulations = number_of_random_simulations
        self.realizations = number_of_realizations
        
        if (self.bins_per_dim/2 == self.bins_per_dim//2+1):
            raise ArgumentError("The number of bins per dimensions must be an odd integer")
        
        self.num_dim = X.shape[1]
        self.multi_dim=multi_dim
        
        maxs = np.max(X,axis=0)
        mins = np.min(X,axis=0)
        widths = (maxs-mins)/(self.bins_per_dim)
        max_lim = maxs.copy()
        min_lim = mins.copy()
        maxs+=int(np.sqrt(self.bins_per_dim))*widths
        mins-=int(np.sqrt(self.bins_per_dim))*widths
        widths = (maxs-mins)/(self.bins_per_dim)
        
        self.dim_bins = [np.linspace(m0,m1,self.bins_per_dim) for m0,m1 in zip(mins,maxs)]
        
        self.dim_combos = [(i,j) for i in range(X.shape[1]) for j in range(X.shape[1]) if j>i]
        #custom dimensional pair TODO
        #if dim_tuples is None:
        #    self.dim_combos = [(i,j) for i in range(X.shape[1]) for j in range(X.shape[1]) if j>i]
        #else:
        #    self.dim_combos=dim_tuples
        
        combo_len = len(self.dim_combos)
        
        _y_ = None
        if self.kde_target is None:
            self.kde_target = np.zeros((combo_len,self.bins_per_dim,self.bins_per_dim),dtype=np.float64)
            self.bin_combos = np.zeros((combo_len,self.bins_per_dim,2),dtype=np.float64)
            count=0
            for i,j in self.dim_combos:
                _X_ = X[:,[i,j]]
                if y is not None and self.multi_dim==True:
                    _y_ = y[:,[i,j]]

                bins3 = [self.dim_bins[i],self.dim_bins[j]]
    #             bins3 = [np.concatenate([[-np.inf],_bins_,[np.inf]]) for _bins_ in bins3]
                self.bin_combos[count] = np.column_stack(bins3)
                data,_ = hdre_histogram(_X_,y,agg_func=agg_func,cost=cost,bins=bins3)
                data=data/np.sum(data)
                self.kde_target[count] = np.fft.fftshift(np.real(np.fft.ifftn(np.fft.fftn(data)*self.fftkernel)))

                count+=1

        
        cycles = 100

        #The number of total simulations per realization = number of cycles * numer of simulations

        zcore = 2.0
        volume = 10 + self.num_dim
        volatility = 1
        zscore = 1


        param_to_dim = np.arange(0,self.num_dim*self.num_clusters)%self.num_dim
        param_to_dim = param_to_dim.astype(np.float)
        self.dim_combos = np.array(self.dim_combos).astype(np.int64)
        args = [self.dim_combos,self.kde_target,self.bin_combos,
                    min_lim,max_lim,param_to_dim,self.num_dim,widths]
        self.args = args
        
        if self.kmldata is None:
            self.kml = KernelML(
                 prior_sampler_fcn=hdre_prior_sampler,
                 posterior_sampler_fcn=None,
                 intermediate_sampler_fcn=None,
                 mini_batch_sampler_fcn=None,
                 parameter_transform_fcn=hdre_parameter_transform,
                 loss_calculation_fcn=_map_losses,
                 batch_size=None)
        else:
            self.kmldata.args = self.args
            self.kml.load_kmldata(self.kmldata)
#            self.kml.kmldata.args = self.args
            
        self.kml.optimize(X[:1],np.array([[]]),
                                        convergence_z_score=3.0,
                                        min_loss_per_change=0.0,
                                        number_of_parameters=self.num_clusters*self.num_dim+self.num_dim,
                                        args=args,
                                        number_of_realizations=self.realizations,
                                        number_of_random_simulations=self.simulations,
                                        update_volume=volume,
                                        update_volatility=volatility,
                                        number_of_cycles=cycles,
                                        print_feedback=verbose)

        self.kmldata = self.kml.kmldata
        self.kml.load_kmldata(self.kmldata)
        
        self.kde_estimate = np.zeros((combo_len,self.bins_per_dim,self.bins_per_dim))
        var = self.deviations_
        mean = self.centroids_

        count=0
        for i,j in self.dim_combos:
            bins = self.bin_combos[count:count+1,:,0].shape[1]
            bins3 = [self.bin_combos[count:count+1,:,0],self.bin_combos[count:count+1,:,1]]
            data1 = np.random.uniform(0,1e-3,size=(bins,bins))
            for k in range(self.num_clusters):

                pdf = uniform_kernel(bins3[0],mean[k,i],var[i,0]).dot(uniform_kernel(bins3[1],mean[k,j],var[j,0]).T)
                data1 += pdf

            data1=data1/np.sum(data1)
#             pdf = np.fft.fftshift(np.real(np.fft.ifftn(np.fft.fftn(data1)*fftkernel)))
            self.kde_estimate[count] = data1
            
#             self.kde_target[count] = np.flipud(self.kde_target[count])
            count+=1
    
    @property
    def deviations_(self):
        w = self.kmldata.best_weight_vector.flatten()
        return np.abs(w[:self.num_dim]).reshape(-1,1)

    @property
    def centroids_(self):
        w = self.kmldata.best_weight_vector.flatten()
        w = w[self.num_dim:]
        return w.reshape((w.size//self.num_dim,self.num_dim))
    
    def prune_clusters(self,X,pad=1.0,limit=0):
        mask=self.get_assignments(X,pad=pad)
        reduce = [i for i in range(mask.shape[1]) if np.sum(mask[:,i]==1,axis=0)<=limit]
        
        
        mean = np.delete(self.centroids_, reduce, axis=0).flatten()
        variance = self.deviations_.flatten()
        
        self.kmldata.best_weight_vector = np.concatenate([variance,mean]).reshape(-1,1)
        
       
        
        centroid_indxs = np.arange(0,self.num_dim*self.num_clusters)
        param_to_clusters = centroid_indxs%self.num_clusters
        self.num_clusters = self.num_clusters - len(reduce)
        self.kmldata.num_clusters = self.num_clusters
    
        reduce = [x+self.num_dim for x in centroid_indxs if param_to_clusters[x] in reduce]
        
        self.kmldata.update_by_cycle = np.delete(self.kmldata.update_by_cycle, reduce, axis=1)
        
        self.kmldata.update_by_realization = np.delete(self.kmldata.update_by_realization, reduce, axis=1)
    
        self.kmldata.update_history = np.delete(self.kmldata.update_history, reduce, axis=0)
        
        self.kmldata.current_weights = np.delete(self.kmldata.current_weights, reduce, axis=0)
        
        self.kmldata.number_of_parameters =self.num_clusters*self.num_dim+self.num_dim
        
        self.get_kde_estimate(pad)
        
    def get_kde_estimate(self,pad=1.0):
        
        dim_combos,pdf_combos,bin_combos,min_lim,max_lim,param_to_dim,num_dim,widths = self.kmldata.args
        combo_len = len(self.dim_combos)

        self.kde_estimate = np.zeros((combo_len,self.bins_per_dim,self.bins_per_dim))
        var = self.deviations_
        mean = self.centroids_
        count=0
        for i,j in self.dim_combos:
            bins = bin_combos[count:count+1,:,0].shape[1]
            bins3 = [bin_combos[count:count+1,:,0],bin_combos[count:count+1,:,1]]
            
            data1 = np.random.uniform(0,1e-3,size=(bins,bins))
            for k in range(self.num_clusters):

                pdf = uniform_kernel(bins3[0],mean[k,i],var[i,0]).dot(uniform_kernel(bins3[1],mean[k,j],var[j,0]).T)
                data1 += pdf

            data1=data1/np.sum(data1)
            self.kde_estimate[count] = data1
            count+=1
    

    def get_polygons(self,i,j,k,pad=1):
        var = self.deviations_
        mean = self.centroids_

        S = np.array([[var[i],0],[0,var[j]]])

        T = (S*pad)

        points = np.array([[-1.,1.],[1.,1.],[1.,-1.],[-1.,-1.],[-1.,1.]])
        points = points.dot(T)
        points[:,0] = points[:,0]+mean[k,i]
        points[:,1] = points[:,1]+mean[k,j]


        return points


    def get_assignments(self,X,pad=1):
        var = self.deviations_
        mean = self.centroids_
        mask = np.zeros((X.shape[0],self.num_clusters),dtype=np.bool)
        count=0
        for k in range(self.num_clusters):
            lower = mean[k].flatten()-var.flatten()*pad
            upper = mean[k].flatten()+var.flatten()*pad

            mask[:,k] = np.all((X>=lower)&(X<=upper),axis=1)

        return mask
    

    def get_distances(self,X,pad=1,distance='chebyshev'):
        var = self.deviations_
        w = self.centroids_

        loss_matrix=np.zeros((X.shape[0],self.num_clusters))
        for i in range(w.shape[0]):
            if distance=='chebyshev':
                loss_matrix[:,i]=np.max(np.abs(X[:]-w[i]),axis=1)
            elif distance=='euclidean':
                loss_matrix[:,i]=np.sum(np.abs(X[:]-w[i])**2,axis=1)
            elif distance=='mae':
                loss_matrix[:,i]=np.mean(np.abs(X[:]-w[i]),axis=1)
            else:
                raise ArgumentError("Invalid distance metric")
        return loss_matrix


class HierarchicalDensityFactorization():
    
    def __init__(self,num_clusters=8,
                 bins_per_dimension=61,
                 smoothing_parameter=1.0,
                 min_leaf_samples=1,
                 agg_func='count',
                 multi_dim=False,
                 alpha = 0.1,
                 verbose=0):
        
        self.alpha = alpha
        self.min_leaf_samples = min_leaf_samples
        self.agg_func = agg_func
        self.multi_dim = multi_dim
        self.assignments = []
        self.centroids = []
        self.deviations = []
        self.min_losses = []
        self.models = []
        self.burn_in = 10
        self.verbose = verbose
        self.kde_true = []
        self.kde_pred = []
        self.KDE = None
        
        self.smoothing_parameter = smoothing_parameter
        
        self.model = DensityFactorization(number_of_clusters=num_clusters,
                                                                    bins_per_dimension=bins_per_dimension,
                                                                    smoothing_parameter=smoothing_parameter)
        self.count = 0
        
        
 
    def refactor_assignments(self,X,pad=1.):
        
        assignments = []
        for model in self.models:
            
            D = model.get_assignments(X,pad=1.).astype(np.int)
            
            Z = np.diag(np.dot(D.T,D))
            for i in range(Z.shape[0]):
                if Z[i]>self.min_leaf_samples:
                    assignments.append(D[:,i:i+1])
                    
#             E = np.hstack(self.assignments)
#             M = (np.sum(E,axis=1)==0)

#             alpha = np.mean(M)*self.alpha

#             self.model.update_kde(X[M],X[M],alpha)
            
        return assignments

#    def assign(self,X,pad=1.):
#        assignments = []
#        for model in self.models:
#
#            D = model.get_assignments(X,pad=1.).astype(np.int)
#
#            Z = np.diag(np.dot(D.T,D))
#            for i in range(Z.shape[0]):
#                if Z[i]>self.min_leaf_samples:
#                    assignments.append(D[:,i:i+1])
#
#        return np.hstack(assignments)
       
    def assign(self,X,pad=1):
        num_clusters = np.vstack(self.centroids).shape[0]
        mask = np.zeros((X.shape[0],num_clusters),dtype=np.bool)
        k=0
        for C,S in zip(self.centroids,self.deviations):
            for i in range(C.shape[0]):
                lower = C[i].flatten()-S.flatten()*pad
                upper = C[i].flatten()+S.flatten()*pad
                mask[:,k] = np.all((X>=lower)&(X<=upper),axis=1)
                k+=1
                
        return mask
    
    def factorize(self,X,M,Y=None):

        
        """
        alpha controls how much weight to give to data points not assign to a cluster
        """
        
        alpha = np.mean(M)*self.alpha

        self.model.update_kde(X[M],X[M],alpha)
        
        realizations = self.realizations
            
        self.model.kmldata=None
        
#         model.prune_clusters(y,pad=2.0,limit=1)
        y=X
        if Y is not None:
            y=Y
            
        
        self.model.optimize(X=X,y=y,  number_of_random_simulations=self.number_of_random_simulations,
                   number_of_realizations = realizations,
                   agg_func=self.agg_func,
                   verbose=self.verbose,
                   multi_dim=self.multi_dim
                     )
        
#         if self.count==0:
#             self.KDE = self.model.kde_target.copy()
        
        self.count+=1
        
        #this will plot the estimated kernel density estimated
        """
        count=0
        for i,j in self.model.dim_combos:

            plt.subplot(1,2,1)
            plt.imshow(self.model.kde_estimate[count])
            plt.title('Estimated KDE')

            plt.subplot(1,2,2)
            plt.imshow(self.model.kde_target[count])
            plt.title('Target KDE')
            plt.show()
            count+=1
        """
            
        self.kde_true.append(self.model.kde_target.copy())
        self.kde_pred.append(self.model.kde_estimate.copy())
        

        D = self.model.get_assignments(X,pad=1.).astype(np.int)
        C = self.model.centroids_
        S = self.model.deviations_

        minloss = self.model.kmldata.loss_by_realization[-1][0]
        self.min_losses.append(minloss)
        
        self.models.append(copy.copy(self.model))
        
        #add data points to the collected list
        Z = np.diag(np.dot(D.T,D))
        
        if self.verbose==True:
            print(np.sum(Z==0),'clusters were not assigned data points\n')
        
        assigns = []
        centers = []
        for i in range(Z.shape[0]):
            #only add if the cluster has new data points?
            if Z[i]>self.min_leaf_samples:
                assigns.append(D[:,i:i+1])
                centers.append(C[i:i+1])
                
                
        if len(centers)>0:
            centers = np.vstack(centers)
            self.assignments.append(assigns)
            self.centroids.append(centers)
            self.deviations.append(S)
            
    def optimize(self,X,Y=None,
                 maxiter=20,
                 stop_thrshld=0.01,
                 realizations=10,
                 number_of_random_simulations=200,
                 dim_tuples=None,
                 verbose=0):
        
        self.verbose = verbose
        self.dim_tuples = dim_tuples
        self.realizations = realizations
        self.number_of_random_simulations = number_of_random_simulations
        M = np.ones(X.shape[0],dtype=bool)
        LAST = M
        for _ in range(maxiter):
            self.factorize(X,M,Y)
            E = np.hstack(sum(self.assignments,[]))
            M = (np.sum(E,axis=1)==0)
            if self.verbose==True:
                print(np.sum(M),'data points are unassigned')
            
            #break when no data points are left
            if np.sum(M)==0:
                break
                
            #break if no threshold is met
                        
            
            prune_losses = []
            prune_kde_true = []
            prune_kde_pred = []
            prune_models = []
            prune_centroids = []
            prune_deviations = []
            prune_assignments = []
            keep_count=0
            
            if len(self.min_losses)<=1:
                continue
            
            
            for i,loss in enumerate(self.min_losses):
                if loss*0.5<=np.min(self.min_losses):
                    prune_models.append(self.models[i])
                    prune_kde_true.append(self.kde_true[i])
                    prune_kde_pred.append(self.kde_pred[i])
                    prune_losses.append(self.min_losses[i])
                    prune_centroids.append(self.centroids[i])
                    prune_deviations.append(self.deviations[i])
                    prune_assignments.append(self.assignments[i])
                    keep_count+=1
                    
            prune_count = len(self.min_losses)-keep_count

            if self.verbose==True and prune_count>0:
                print('pruning {} iterations\n'.format(prune_count))
                
            self.models = prune_models
            self.kde_true = prune_kde_true
            self.kde_pred = prune_kde_pred
            self.min_losses = prune_losses
            self.centroids = prune_centroids
            self.deviations = prune_deviations
            self.assignments = prune_assignments
            

#            if prune_count>0:
#                self.assignments = self.refactor_assignments(X)
            
            
            
            if np.mean(M)<stop_thrshld and prune_count==0:
                if self.verbose==True:
                    print('stopping with {} unassigned data points'.format(np.sum(M)))
                break
            
            if (np.sum(LAST)-np.sum(M))==0 and prune_count==0:
                if self.verbose==True:
                    print('stopping with {} unassigned data points'.format(np.sum(M)))
                break
                
            LAST = M
            
        if self.verbose==True and self.count==maxiter:
            print('stopping at {} iterations'.format(maxiter))
        

            
        
            

@jit('float64[:,:](float64[:,:], float64,float64)',nopython=True)
def uniform_kernel(x,mu_,var_):

    mu = mu_
    
    x = x.ravel()

    width = x[1]-x[0]
    var = var_
    
    if var<width:
        var = width
    
    pdf = ((x+width>=mu-var)&(x+width<=mu+var)).reshape(-1,1)
    pdf = pdf.astype(np.float64)
    
    if x[1]>mu-var:
        pdf[0] = pdf[0] + var*2/width - np.sum(pdf)
    if x[-2]<mu+var:
        pdf[-1] = pdf[-1] + var*2/width - np.sum(pdf)
    result = (pdf/np.sum(pdf))
    return result



@jit('float64(float64[:,:], float64[:,:], float64[:,:],int64[:,:],float64[:,:,:],float64[:,:,:],float64[:],float64[:],float64[:],int64,float64[:])',nopython=True)
def hdre_loss(X,y,w_,dim_combos, pdf_combos,bin_combos, min_lim, max_lim, param_to_dim, num_dim, widths):
     var =np.copy(np.abs(w_[:num_dim])).astype(np.float64)

     num_clusters = int(w_[num_dim:].size/num_dim)

     samples = 100
     w =  np.copy(w_[num_dim:]).reshape((num_clusters,num_dim)).astype(np.float64)

 #     assert num_clusters==w.size/dim

     N = dim_combos.shape[0]

     bin_combos2 = np.copy(bin_combos).astype(np.float64)
     ll=0
     
     count=0
     for n in range(N):
         i,j = dim_combos[n]
         binsx = bin_combos[count:count+1,:,0].shape[1]
         bins = bin_combos2[count:count+1,:,:].astype(np.float64)
         
         data1 = np.random.uniform(0,1e-3,size=(binsx,binsx))
         for k in range(w.shape[0]):
             bins0 = bins[:,:,0].astype(np.float64)
             bins1 = bins[:,:,1].astype(np.float64)
 #             uniform_kernel(bins0,w[k,i],var[i,0])
             pdf = uniform_kernel(bins0,w[k,i],var[i,0]).dot(uniform_kernel(bins1,w[k,j],var[j,0]).T)
          
 #             print(pdf.shape)
             data1 += pdf
             

         pdf1=data1/np.sum(data1)

         pdf2 = pdf_combos[count]

         diff = (pdf2-pdf1).flatten()*100
         
         ll += np.sum(diff**2)

         count+=1

     return ll

@njit('float64[:](float64[:,:], float64[:,:], float64[:,:],int64[:,:],float64[:,:,:],float64[:,:,:],float64[:],float64[:],float64[:],int64,float64[:])',parallel=True)
def _map_losses(X, y, w_list, dim_combos, pdf_combos,bin_combos, min_lim, max_lim, param_to_dim, num_dim, widths):
    N = w_list.shape[1]
    resX = np.zeros(N)
    for i in prange(N):
        loss = hdre_loss(X,y,w_list[:,i:i+1],dim_combos, pdf_combos,bin_combos, min_lim, max_lim, param_to_dim, num_dim, widths)
        resX[i] = loss
    return resX

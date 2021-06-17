# cython: language_level = 3


import time
import numpy as np
import numpy

class ArgumentError(Exception):
    pass

def hdre_prior_sampler(kmldata):
    random_samples = kmldata.prior_random_samples
    lows,highs,_,num_dim,_ = kmldata.args[-5:]
    parts = len(lows)
    num_params = kmldata.number_of_parameters-num_dim
    #     print(num_params,parts)
    sizes = np.linspace(0,num_params,(parts+1))[1:]/np.arange(0,parts+1)[1:]
    sizes[-1] = num_params-np.sum(sizes[:-1].astype(np.int))
    #     sizes[:-1] = sizes[:-1]
    #     print(sizes)
    sizes = sizes.astype(np.int)

    drange = highs-lows
    output = []
    for i in range(num_dim):
        output.append(np.random.uniform(0,drange[i],random_samples))
    output = np.vstack(output)
    i = 0
    j = 0
    while i<num_params:
        low = lows[j]
        high = highs[j]
        size = sizes[j]
        r = np.random.uniform(low=low,high=high,size=(size,random_samples))
        output = np.vstack([output,r])
        j+=1
        i+=num_params//parts
    return output



def hdre_parameter_transform(w,dim_combos, pdf_combos,bin_combos, min_lim, max_lim, param_to_dim, num_dim, widths):

    v = np.abs(w[:num_dim])
    drange = (max_lim-min_lim)*0.5
    for i in range(widths.shape[0]):
        v[i][v[i]<(widths[i])] = (widths[i])
        v[i][v[i]>(drange[i])] = (drange[i])

    w[:num_dim] = v
    p = w[num_dim:]

    count=0
    for low,high in zip(min_lim,max_lim):
        q = p[param_to_dim==count]
        q[q<low] = low
        q[q>high] = high
        p[param_to_dim==count]=q
        count+=1

    w[num_dim:]=p
    return w


def hdre_histogram(X, y = None, cost=None, agg_func='count',bins=10):

    _has_target_ = False
    _multi_dim_ = False
    


    if y is not None:
        _has_target_ = True

    if cost is None:
        cost = np.ones(y.shape[0])

    if X.shape[1]==2:
        X1,X2 = np.split(X,2,axis=1)

    if agg_func == 'entropy':
        assert y.shape[1]==2

    if isinstance(bins, (int)):
        _bins_ = []

        _bins_ += [np.linspace(np.min(X1),np.max(X1),bins)]
        _bins_ += [np.linspace(np.min(X2),np.max(X2),bins)]

    else:
        _bins_ = bins
        
    if agg_func=='count':
    
        _bins_0 = np.concatenate([_bins_[0],[_bins_[0][-1]+np.diff(_bins_[0])[0]]])
        _bins_1 = np.concatenate([_bins_[1],[_bins_[1][-1]+np.diff(_bins_[1])[0]]])
        
        data = np.histogram2d(X1.flatten(),X2.flatten(),bins=[_bins_0,_bins_1])
        _bins_ = data[1]
        heatmap = data[0]
        return heatmap,_bins_


    _num_bins_1 = len(_bins_[0])
    _num_bins_2 = len(_bins_[1])


    heatmap = np.zeros((_num_bins_1,_num_bins_2))


    X1 = np.digitize(X1,_bins_[0])-1

    X2 = np.digitize(X2,_bins_[1])-1

    for i in range(_num_bins_1):
        for j in range(_num_bins_2):
            count = np.sum((X1==i)&(X2==j))
            if _has_target_==True and agg_func=='mean' and count>0:
                loc = ((X1==i)&(X2==j)).flatten()
                heatmap[i,j]=np.mean(y[loc])
            elif _has_target_==True and agg_func=='variance' and count>1:
                loc = ((X1==i)&(X2==j)).flatten()
                heatmap[i,j]=np.var(y[loc])
            elif _has_target_==True and agg_func=='max' and count>0:
                loc = ((X1==i)&(X2==j)).flatten()
                heatmap[i,j]=np.max(y[loc])
            elif _has_target_==True and agg_func=='false-negative-cost' and count>0:
                loc = ((X1<=i)&(X2<=j)).flatten()
                heatmap[i,j]=np.mean(1-y[loc])/np.sum(cost[loc])
            elif _has_target_==True and agg_func=='false-positive-cost' and count>0:
                loc = ((X1>=i)&(X2>=j)).flatten()
                heatmap[i,j]=np.mean(y[loc])/np.sum(cost[loc])
            elif _has_target_==True and agg_func=='entropy' and count>0:
                loc = ((X1==i)&(X2==j)).flatten()
                _y_ = y[loc]
                for d in range(_y_.shape[1]):
                    heatmap[i,j]+= - np.mean(_y_[:,d])*np.log(np.mean(_y_[:,d]))
            elif agg_func=='count':
                loc = ((X1==i)&(X2==j)).flatten()
                heatmap[i,j]=count
#else:
#               raise ArgumentError("Argument '{}' is not valid".format(agg_func))

    return heatmap,_bins_

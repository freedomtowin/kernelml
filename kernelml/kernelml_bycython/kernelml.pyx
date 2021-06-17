# cython: language_level = 3

import time
import numpy as np
#import matplotlib.pyplot as plt
import heapq
import gc


class CustomException(Exception):
    pass

class kernel_optimizer():

    def __init__(self,X,y,loss_function,number_of_parameters,args=[],model=None):
        self.X = X
        self.y = y
        self.loss_calculation = loss_function
        self.kmldata = model
        self.number_of_parameters = number_of_parameters
        self.args = np.array(args)
        self.loss_by_cycle = []
        self.update_by_cycle = []
        self.default_random_simulation_params()
        self.adjust_optimizer()
        self.prior_sampler = self.prior_sampler_uniform_distribution
        self.random_sampler = self.posterior_sampler_uniform_distribution
        self.parameter_transform = self.default_parameter_transform
        self.convergence_z_score = 2
        self.min_loss_per_change = 0
        self.has_args = False


        if len(self.args)>0:
            self.has_args = True


    def set_weights(self,w):
        self.kmldata.current_weights = w

    def heapsort(self,iterable):
        h = []
        for value in iterable:
            heapq.heappush(h, value)
        return [heapq.heappop(h) for i in range(len(h))]

    def adjust_convergence_z_score(self,z):
        self.convergence_z_score = z

    def adjust_min_loss_per_change(self,mlpc):
        self.min_loss_per_change = mlpc

    def default_parameter_transform(self,w,*args):
        return w

    def default_random_simulation_params(self,prior_random_samples=None,random_sample_num=None,prior_uniform_low=-1,prior_uniform_high=1):
        self.kmldata.prior_random_samples = prior_random_samples
        self.kmldata.posterior_random_samples = random_sample_num
        self.low = prior_uniform_low
        self.high = prior_uniform_high

    def adjust_optimizer(self,number_of_random_simulations=100,update_volatility=1,number_of_cycles=10,
                                   update_volume=2,
                                   analyze_n_parameters=None
                                   ):

        self.number_of_random_simulations = number_of_random_simulations
        self.update_volatility = update_volatility
        self.number_of_cycles = number_of_cycles
        self.update_volume = update_volume
        self.strokes = np.minimum(update_volume,100)

        self.analyze_n_parameters = analyze_n_parameters
        if self.analyze_n_parameters is None:
            self.analyze_n_parameters =  int(0.25*self.number_of_random_simulations)

        if self.kmldata.prior_random_samples is None:
            self.kmldata.prior_random_samples = int(self.number_of_random_simulations)

        if self.kmldata.posterior_random_samples is None:
            self.kmldata.posterior_random_samples = int(self.number_of_random_simulations)


        self.P = 0.9
        self.I = 0.9
        self.D = 0

    def _update_parameter_history(self):

        if len(self.kmldata.update_history) == 0:
            self.kmldata.update_history = self.kmldata.current_weights
            self.kmldata.loss_history = self.loss
            return
        try:
            self.kmldata.update_history = np.column_stack((self.kmldata.update_history,self.kmldata.current_weights))
        except Exception as e:
                raise CustomException("Not enough samples, try increasing the number of simulations or volatility")

        self.kmldata.loss_history =  np.hstack((self.kmldata.loss_history,self.loss))

        try:
            self.kmldata.update_history.shape[1]
            self.kmldata.update_history.shape[0]
        except Exception as e:
            raise CustomException('parameter input must be 2-D, please check sample size or number of parameters.')


    def _sort_parameters_on_loss(self):
        self.kmldata.current_weights = self.parameter_transform(self.kmldata.current_weights.copy(),*self.args)

        tmp_loss =self.loss_calculation(self.X,self.y,self.kmldata.current_weights.copy(),*self.args)
    
        ar = np.arange(tmp_loss.shape[0])
            
        heap = [(l,i) for l,i in zip(tmp_loss,ar) if np.abs(l)<np.float('inf') and ~np.isnan(l)]
        
        sorted_heap = self.heapsort(heap)
        
        
        self.kmldata.current_weights = np.array([self.kmldata.current_weights[:,i]
                                for loss,i in reversed(sorted_heap[:self.analyze_n_parameters])]).T
        
        self.loss= np.array([loss
                                for loss,i in reversed(sorted_heap[:self.analyze_n_parameters])])
        
        self._update_parameter_history()


    def prior_sampler_uniform_distribution(self,kmldata):
        random_samples = kmldata.prior_random_samples
        number_of_parameterss = kmldata.number_of_parameters
        return np.random.uniform(low=self.low,high=self.high,size=(number_of_parameterss,
                                                                   random_samples))

    def posterior_sampler_uniform_distribution(self,kmldata):

        random_samples = kmldata.prior_random_samples
        variances = np.var(kmldata.update_history[:,:],axis=1).flatten()
        means = kmldata.best_weight_vector.flatten()

        return np.vstack([np.random.uniform(mu-np.sqrt(sigma*12)/2,mu+np.sqrt(sigma*12)/2,(random_samples)) for sigma,mu in zip(variances,means)])



    def _cycle(self, wh, lh, P=0.1, I=0.1, D=0):


        w = self.kmldata.update_history[:,-1:]

        sw = (self.kmldata.loss_history-np.mean(self.kmldata.loss_history))/np.std(self.kmldata.loss_history)
        sw = np.clip(sw,-123.1,123.1)
        sw = (np.exp(-1*sw)/(1+np.exp(-1*sw)))*self.update_volatility

        sw = np.nan_to_num(sw).reshape(-1,1)
        
        if np.sum(sw)==0:
            raise CustomException("Not enough variation in sampled parameters, try increasing the number of simulations or volatility")

        _y_ = self.kmldata.loss_history.reshape(-1,1)
        _x_ = self.kmldata.update_history.T
        #_x_ = _x_ - np.mean(_x_,axis=0)
        _x_ = np.column_stack((np.ones((_x_.shape[0],1)),_x_))*np.sqrt(sw)
        A=_x_.T.dot(_x_)
        b=_x_.T.dot(_y_*np.sqrt(sw))
        
        choices = np.arange(0,w.shape[0])
        
        try:
            _z_ = np.linalg.solve(A,b)
            coefs = _z_[1:]
            effect_size = np.abs(coefs.flatten()*w.flatten())
        except:
            coefs = np.std(self.kmldata.update_history[:,:],axis=1).flatten()
            effect_size =  np.std(self.kmldata.update_history[:,:],axis=1).flatten()
            
            
        norm_effect_size = effect_size/np.sum(effect_size)
        best_n_effects = np.random.choice(choices, np.minimum(self.update_volume,w.shape[0]), p=norm_effect_size,replace=False)

        
        #best_n_effects  = np.argsort(effect_size)[-self.update_volume:]
        mutation_list = np.concatenate([w for _ in range(best_n_effects.shape[0])],axis=1)[:,:,np.newaxis]
        mutation_list = np.concatenate([mutation_list for _ in range(self.strokes)],axis=2)

        count=0

        for i_ in best_n_effects:

            w_tmp  = w.copy()
            pid = parameter_search(P,I,D)

            pid.SetPoint = self.kmldata.update_history[i_][-1]
            feedback=-self.update_volatility*coefs[i_]+self.kmldata.update_history[i_][-1]

            pid.windup_guard = np.exp(np.clip(coefs[i_]*self.kmldata.update_history[i_][-1] /np.abs(self.kmldata.update_history[i_][-1]),-123.1,123.1))

            updates = np.zeros(self.strokes)
            for p in range(0,self.strokes):
                updates[p] = feedback
                pid.update(feedback)
                output = pid.output
                feedback += output


            mutation_list[i_,count,:] = updates

            count+=1


        mutation_list = self.parameter_transform(mutation_list,*self.args)
        successful_mutations = self._multiple_fire(mutation_list)
        self._breed(successful_mutations)

    def _multiple_fire(self,mutation_list):


        w = self.kmldata.update_history[:,-1].reshape((self.kmldata.update_history.shape[0],1))
        loss_1 = self.kmldata.loss_history[-1]
        for p in range(0,self.strokes):
            misfire=True

            tmp_loss =self.loss_calculation(self.X,self.y,mutation_list[:,:,p],*self.args)
        
            ar = np.arange(tmp_loss.shape[0])

            heap = [(l,i) for l,i in zip(tmp_loss,ar) if np.abs(l)<np.float('inf') and ~np.isnan(l)]
        
            sorted_heap = reversed(self.heapsort(heap))
            
            reduced_heap = [(loss_2,i) for loss_2,i in sorted_heap if (loss_2<loss_1) and abs(loss_2)<np.float('inf')]
            n_succesful_mutations = len(reduced_heap)
            successful_mutations = np.zeros((w.shape[0],n_succesful_mutations))
            count = 0
            for loss_2,i_ in reduced_heap:

                w = mutation_list[:,i_,p:p+1]
                self.kmldata.update_history = np.column_stack((self.kmldata.update_history,w))
                self.kmldata.update_history = self.kmldata.update_history[:,1:]

                self.kmldata.loss_history = np.hstack((self.kmldata.loss_history,np.array([loss_2])))
                self.kmldata.loss_history = self.kmldata.loss_history[1:]
                loss_1=loss_2
                misfire=False

                successful_mutations[:,count:count+1] = w
                count+=1
                self.kmldata.increment_number_of_updates()

            if misfire==False:
                break


        return successful_mutations


    def _breed(self,successful_mutations):

        N = successful_mutations.shape[1]
        if N<=1:
            return 0
        loss_1 = self.kmldata.loss_history[-1]
        n_breed_combos = (N+1)*N//2 - N
        breed_list = np.zeros((successful_mutations.shape[0],n_breed_combos))
        count = 0
        for i in range(successful_mutations.shape[1]):
            for j in range(successful_mutations.shape[1]):
                if i<=j:
                    continue
                breed_list[:,count] =0.5*successful_mutations[:,i]+0.5*successful_mutations[:,j]
                count+=1

        breed_list = self.parameter_transform(breed_list,*self.args)

        tmp_loss =self.loss_calculation(self.X,self.y,breed_list,*self.args)
                
        ar = np.arange(tmp_loss.shape[0])
        
        heap = [(l,i) for l,i in zip(tmp_loss,ar) if np.abs(l)<np.float('inf') and ~np.isnan(l)]

        sorted_heap = reversed(self.heapsort(heap))
        
        reduced_heap = [(loss_2,i) for loss_2,i in sorted_heap if (loss_2<loss_1) and abs(loss_2)<np.float('inf')]
        for loss_2,i in reduced_heap:

            w = breed_list[:,i:i+1]
            self.kmldata.update_history = np.column_stack((self.kmldata.update_history,w))
            self.kmldata.update_history = self.kmldata.update_history[:,1:]

            self.kmldata.loss_history = np.hstack((self.kmldata.loss_history,np.array([loss_2])))
            self.kmldata.loss_history = self.kmldata.loss_history[1:]
            self.kmldata.increment_number_of_updates()

    def optimize(self,print_feedback=False):

        self.kmldata.current_weights = self.prior_sampler(self.kmldata)
        self.kmldata.reset_histories()
        convergence = np.ones(self.kmldata.number_of_parameters)*3
        loss_per_change = 1e6

        proportional = np.linspace(0.0,0.2,self.update_volume)
        integral = np.linspace(0.0,0.2,self.update_volume)

        for n in range(0,self.number_of_cycles):

            self._sort_parameters_on_loss()

            self._cycle(                           self.kmldata.update_history[:,:],
                                                  self.kmldata.loss_history[:],
                                                  self.P,
                                                  self.I,
                                                  self.D)


            self.loss_by_cycle.append(self.kmldata.loss_history[-1])
            self.update_by_cycle.append(self.kmldata.update_history[:,-1])

            self.kmldata.update_by_cycle = np.array(self.update_by_cycle)
            self.kmldata.loss_by_cycle = np.array(self.loss_by_cycle)
            self.kmldata.best_weight_vector = self.kmldata.update_by_cycle[np.where(self.kmldata.loss_by_cycle==np.min(self.kmldata.loss_by_cycle))[0]]


            if n>0:
                loss_per_change = np.abs((self.kmldata.loss_by_cycle[-1]-self.kmldata.loss_by_cycle[-2])/self.kmldata.loss_by_cycle[-2])

            if loss_per_change<self.min_loss_per_change:
                return True

            #current only accepts parameter vector
            if self.kmldata.best_weight_vector.shape[0]>1:
                self.kmldata.best_weight_vector = self.kmldata.best_weight_vector[0]

            if n>=10:
                convergence = (self.kmldata.best_weight_vector-np.mean(
                                self.kmldata.update_by_cycle[-10:,:],axis=0))/(np.std(
                                self.kmldata.update_by_cycle[-10:,:],axis=0))
                nans = np.isnan(convergence)
                convergence[nans] = 0
                convergence = np.abs(convergence)

            if np.all(np.abs(convergence)<self.convergence_z_score):
                return True


            new_weights = self.random_sampler(self.kmldata)

            if new_weights is not None:
                self.kmldata.current_weights = new_weights

            self.kmldata.increment_cycle_number()
            gc.collect()

        return False

    def change_random_sampler(self,fcn):
        self.random_sampler = fcn

    def change_prior_sampler(self,fcn):
        self.prior_sampler = fcn

    def change_parameter_transform(self,fcn):
        self.parameter_transform = fcn

    def change_loss_calculation(self,fcn):
        self.loss_calculation = fcn

    def get_param_by_iter(self):
        return np.array(self.kmldata.update_by_cycle)

    def get_loss_by_iter(self):
        return np.array(self.kmldata.loss_by_cycle)

    def get_best_param(self):
        params = self.get_param_by_iter()
        errors = self.get_loss_by_iter()
        best_w_arr = errors.argsort()[0]
        w = params[best_w_arr]
        return w

    def get_update_history(self):
        return self.kmldata.update_history


class parameter_search:

    def __init__(self, P=0.99, I=0.0, D=0.0):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = 0.0000
        self.last_time = self.current_time

        self.clear()

    def clear(self):

        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 1

        self.output = 0.0

    def update(self, feedback_value):

        error = self.SetPoint - feedback_value

        self.current_time = self.current_time+0.0005
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        self.Kd = derivative_gain

    def setWindup(self, windup):
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        self.sample_time = sample_time

class KernelML():

    def __init__(self,
                 loss_calculation_fcn,
                 prior_sampler_fcn=None,
                 posterior_sampler_fcn=None,
                 intermediate_sampler_fcn=None,
                 mini_batch_sampler_fcn=None,
                 parameter_transform_fcn=None,
                 batch_size=None):

        self.loss_calculation_fcn = loss_calculation_fcn
        self.prior_sampler_fcn = prior_sampler_fcn
        self.posterior_sampler_fcn = posterior_sampler_fcn
        self.intermediate_sampler_fcn = self.intermediate_sampler_uniform_distribution
        if intermediate_sampler_fcn is not None:
            self.intermediate_sampler_fcn = intermediate_sampler_fcn
        self.mini_batch_sampler_fcn = self.mini_batch_random_choice
        if mini_batch_sampler_fcn is not None:
            self.mini_batch_sampler_fcn = mini_batch_sampler_fcn
        self.parameter_transform_fcn = parameter_transform_fcn
        self.w = []
        self.batch_size=batch_size
        self.ipyparallel=False
        self.kmldata=None

    class KMLData():

        def __init__(self,prior_random_samples,posterior_random_samples,number_of_parameters,args):
            self.current_weights = np.array([])
            self.best_weight_vector = np.array([])
            self.loss_history = []
            self.update_history = []
            self.update_by_cycle = []
            self.loss_by_cycle = []
            self.loss_by_realization = []
            self.update_by_realization = []
            self.realization_number = 1
            self.cycle_number = 1
            self.number_of_updates = 1
            self.args = args
            self.prior_random_samples = prior_random_samples
            self.posterior_random_samples = posterior_random_samples
            self.number_of_parameters = number_of_parameters

        def increment_realization_number(self):
            self.realization_number = self.realization_number+1

        def increment_cycle_number(self):
            self.cycle_number = self.cycle_number+1

        def increment_number_of_updates(self):
            self.number_of_updates = self.number_of_updates+1

        def reset_histories(self):
            self.loss_history = []
            self.update_history = []
            self.cycle_number=1


    def load_kmldata(self,kmldata):
        self.kmldata = kmldata

    def intermediate_sampler_uniform_distribution(self,kmldata):

        random_samples = kmldata.prior_random_samples
        variances = np.var(kmldata.update_history[:,:],axis=1).flatten()
        means = kmldata.update_history[:,-1].flatten()

        return np.vstack([np.random.uniform(mu-np.sqrt(sigma*12)/4,mu+np.sqrt(sigma*12)/4,(random_samples)) for sigma,mu in zip(variances,means)])

    def mini_batch_random_choice(self,X,y,batch_size):
        all_samples = np.arange(0,X.shape[0])
        rand_sample = np.random.choice(all_samples,size=batch_size,replace=False)
        X_batch = X[rand_sample]
        y_batch = y[rand_sample]
        return X_batch,y_batch

    def optimize(self,X,y,number_of_parameters,args=[],
            number_of_realizations=1,
            number_of_cycles=20,
            update_volume=10,
            number_of_random_simulations = 1000,
            update_volatility = 1,
            min_loss_per_change=0,
            convergence_z_score=1,
            analyze_n_parameters=None,
            update_magnitude=None,
            prior_random_samples=None,
            posterior_random_samples=None,
            prior_uniform_low=-1,
            prior_uniform_high=1,
            print_feedback=True):

        fresh_start = False
        if self.kmldata is None:
            fresh_start = True
            self.kmldata = self.KMLData(prior_random_samples,posterior_random_samples,number_of_parameters,args)

        parameters = np.zeros((number_of_realizations,number_of_parameters))
        losses = np.zeros((number_of_realizations,1))
        for run in range(number_of_realizations):

            start_time = time.time()
            if self.batch_size is None:
                X_batch = X
                y_batch = y
            else:
                X_batch,y_batch = self.mini_batch_sampler_fcn(X,y,self.batch_size)

            self.model = kernel_optimizer(X_batch,y_batch,self.loss_calculation_fcn,
                                      number_of_parameters=number_of_parameters,
                                      args=args,model=self.kmldata)


            #random simulation parameters
            self.model.default_random_simulation_params(prior_uniform_low=prior_uniform_low
                                                   ,prior_uniform_high=prior_uniform_high
                                                   ,prior_random_samples=prior_random_samples
                                                   ,random_sample_num=posterior_random_samples)


            self.model.adjust_optimizer(number_of_cycles=number_of_cycles,
                                            number_of_random_simulations=number_of_random_simulations,
                                                update_volatility=update_volatility,
                                                analyze_n_parameters=analyze_n_parameters,
                                                update_volume=update_volume)

            if self.ipyparallel:
                self.model.init_ipyparallel(self.dview)
            if self.prior_sampler_fcn is not None:
                self.model.change_prior_sampler(self.prior_sampler_fcn)
            if self.posterior_sampler_fcn is not None:
                self.model.change_random_sampler(self.posterior_sampler_fcn)
            if self.parameter_transform_fcn is not None:
                self.model.change_parameter_transform(self.parameter_transform_fcn)

            self.model.adjust_convergence_z_score(convergence_z_score)
            self.model.adjust_min_loss_per_change(min_loss_per_change)

            if run==0 and fresh_start==False:
                self.model.change_prior_sampler(self.intermediate_sampler_fcn)

            if run>0:
                #this overrides the default simulation function
                #we will use the best parameter set from the previous run as the priors for the current
                self.kmldata = self.model.kmldata
                self.model.change_prior_sampler(self.intermediate_sampler_fcn)

            #fix plot_feedback
            self.model.optimize(print_feedback=print_feedback)

            params = self.model.get_param_by_iter()
            errors = self.model.get_loss_by_iter()
            best_w_arr = errors.argsort()[0]
            loss = errors[best_w_arr]
            w = params[best_w_arr].T

            self.w = w
            parameters[run] = w
            losses[run] = loss
            self.kmldata.increment_realization_number()
            end_time = time.time()

            if print_feedback:
                print('realization',run,'loss',loss,'time',end_time-start_time)

        self.kmldata.loss_by_realization = losses
        self.kmldata.update_by_realization = parameters
        return self




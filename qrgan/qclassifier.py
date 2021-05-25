# ###############################################################################
#
# File qgenerator.py
#
# Copyright (C) 2021 Anthony Francis
#
# This software is distributed under the terms of the GNU General Public
# License (GPL)
#
# Routines to run the QML quantum generator: 
# 
# Code based on qibo qlassifier tutorial found at 
# https://qibo.readthedocs.io/en/stable/tutorials/reuploading_classifier/README.html
# Authors: Adrian Perez Salinas, Stefano Carrazza, Stavros Efthymiou
# Distributed under the apache 2 license.  
#
# ###############################################################################

from qibo.models import Circuit
from qibo import hamiltonians, gates, models
import numpy as np
from datasets import create_training
import tensorflow as tf
import os
import qibo as qibo
from itertools import product


# set the number of threads to 1, if you want to
qibo.set_threads(1)
# retrieve the current number of threads
current_threads = qibo.get_threads()
print("# Qibo runs with "+str(current_threads)+" thread(s)")


class single_qubit_classifier:
    def __init__(self, layers, grid=None, seed=0):
       
       
        print('# Setting up quantum classifier ...')
        
        np.random.seed(seed)
        self.layers = layers
                
        self.params = np.random.randn(layers * 4)
        self._circuit = self._initialize_circuit() # initialise with random parameters
                    

    def set_parameters(self, new_params):
        """Method for updating parameters of the class.
        Args:
            new_params (array): New parameters to update
        """
        self.params = new_params
        
    # generator circuit  
    def _initialize_circuit(self):
        """Creates variational circuit."""
        C = Circuit(1)
        for l in range(self.layers):
            C.add(gates.RY(0, theta=0))
            C.add(gates.RZ(0, theta=0))
        return C            

    def circuit(self, x):
        """Method creating the circuit for a point (in the datasets).
        Args:
            x (array): Point to create the circuit.
        Returns:
            Qibo circuit.
        """
        params = []
        for i in range(0, 4 * self.layers, 4):
            params.append(self.params[i] * x + self.params[i + 1]) # x is scalar in this case
            params.append(self.params[i + 2] * x + self.params[i + 3])  # x is scalar in this case
        self._circuit.set_parameters(params)
        return self._circuit
  
    def set_data(self, xval, yval):
        self.data = xval
        self.labl = yval
        #print(self.data,self.labl)

    def set_fake(self, xval, yval):
        self.fake = xval
        self.fabl = yval
        #print(self.data,self.labl)


    # discriminator cost function needs both real and fake data                  
    def cost_function(self, params=None):
        
        # need a blank state to contract fidelity against
        blank_state = [np.array([1, 0], dtype='complex'), np.array([0, 1], dtype='complex')]
        
        # setup parameters
        if params is None:
            params = self.params       

        self.set_parameters(params)                  
          
        # real data component            
        tots=len(self.data[0])
        cf1=0
        i=0
        for x in self.data[0]:
            
            # set label
            y=self.labl[0][i]
            i+=1
                
            # generate the output from our circuit     
            C = self.circuit(x)
            state1 = C.execute()
            
            # associate cost
            cf1 += .5 * (1 - fidelity(state1, blank_state[int(y)])) ** 2
         
        cf1 /= tots  
        
        # fake data component
        cf2=0    
        i=0    
        for z in self.fake[0]:
            
            # set label
            w=self.fabl[0][i]
            i+=1
                
            # generate the output from our circuit     
            C = self.circuit(z)
            state1 = C.execute()
            
            # associate cost
            cf2 += .5 * (1 - fidelity(state1, blank_state[int(w)])) ** 2
                      
        cf2 /= tots 
        
        
        cf=0.5*(cf1+cf2)
        
        tflabel = tf.convert_to_tensor(cf, dtype=tf.float64)
        cf=(tflabel)
        
        return cf    
        
    

    def minimize(self, method='BFGS', options=None, compile=True):
        loss = self.cost_function
        #print("# Run minimisation:")
        
        if method == 'cma':
            # Genetic optimizer
            import cma
            #r = cma.fmin2(lambda p: loss(p).numpy(), self.params, 2)
            #r = cma.fmin2(lambda p: loss(p).numpy(), self.params, 2, {'seed':113895, 'maxiter': 2})
            r = cma.fmin2(lambda p: loss(p).numpy(), self.params, 2, options=options)
            result = r[1].result.fbest
            parameters = r[1].result.xbest

        elif method == 'sgd':
            from qibo.tensorflow.gates import TensorflowGate
            circuit = self.circuit(self.data_set[0])
            for gate in circuit.queue:
                if not isinstance(gate, TensorflowGate):
                    raise RuntimeError('SGD VQE requires native Tensorflow '
                                       'gates because gradients are not '
                                       'supported in the custom kernels.')

            sgd_options = {"nepochs": 5001,
                           "nmessage": 1000,
                           "optimizer": "Adamax",
                           "learning_rate": 0.5}
            if options is not None:
                sgd_options.update(options)

            # proceed with the training
            from qibo.config import K
            vparams = K.Variable(self.params)
            optimizer = getattr(K.optimizers, sgd_options["optimizer"])(
                learning_rate=sgd_options["learning_rate"])

            def opt_step():
                with K.GradientTape() as tape:
                    l = loss(vparams)
                grads = tape.gradient(l, [vparams])
                optimizer.apply_gradients(zip(grads, [vparams]))
                return l, vparams

            if compile:
                opt_step = K.function(opt_step)

            l_optimal, params_optimal = 10, self.params
            for e in range(sgd_options["nepochs"]):
                l, vparams = opt_step()
                if l < l_optimal:
                    l_optimal, params_optimal = l, vparams
                if e % sgd_options["nmessage"] == 0:
                    print('ite %d : loss %f' % (e, l.numpy()))

            result = self.cost_function(params_optimal).numpy()
            parameters = params_optimal.numpy()

        else:
            import numpy as np
            from scipy.optimize import minimize
            m = minimize(lambda p: loss(p).numpy(), self.params, method=method, options=options)
            result = m.fun
            parameters = m.x

        return result, parameters



    def predict(self, xval, params=None):
        
        if params is None:
            params = self.params       

        self.set_parameters(params)        
             
        # need a blank state to contract fidelity against
        blank_state = [np.array([1, 0], dtype='complex'), np.array([0, 1], dtype='complex')]
        
        labels = [[0]] * len(xval[0])
        for j, x in enumerate(xval[0]):
            C = self.circuit(x)
            state = C.execute()
            fids = np.empty(len(blank_state))
            for i, t in enumerate(blank_state):
                fids[i] = fidelity(state, t)
            labels[j] = float(np.argmax(fids))

        #print(labels)
        return labels
            
        


def qgen_real_out(state):
    hx  = hamiltonians.X(1, numpy=True) # create a 1-qubit Pauli-X Hamiltonian
    num = hx.expectation(state).numpy().real # project the state (output from generator) with Hamiltonian 
    #res = (1 - num) / (1+num) # relate this way, it's positive
    #res = np.abs(num) # make it easier for the network by mapping the result between 0 and 1 
    res = num # between -1 and 1
    return res
         
def fidelity(state1, state2):
    return tf.constant(tf.abs(tf.reduce_sum(tf.math.conj(state2) * state1))**2)

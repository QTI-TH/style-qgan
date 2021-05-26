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
from datasets import create_dataset
import tensorflow as tf
import os
import qibo as qibo
from itertools import product
from scipy.stats import entropy
from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp


# set the number of threads to 1, if you want to
qibo.set_threads(1)
# retrieve the current number of threads
current_threads = qibo.get_threads()
print("# Qibo runs with "+str(current_threads)+" thread(s)")


class single_qubit_generator:
    def __init__(self, glayers, dlayers, nmeas=10, grid=None, seed=0):
       
       
        print('# Setting up quantum generator...')
         
        np.random.seed(seed)
        self.glayers = glayers
        self.dlayers = dlayers
        self.nmeas = nmeas
                     
        self.params = np.random.randn(glayers * 4)
        self._circuit = self._initialize_circuit() # initialise with random parameters
 
        self.dparams = np.random.randn(dlayers * 4)
        self._dcircuit = self._initialize_dcircuit() # initialise with random parameters
                    
                    
                    
    # generator circuit      
    def set_parameters(self, new_params):
        """Method for updating parameters of the class.
        Args:
            new_params (array): New parameters to update
        """
        self.params = new_params       
    
    def _initialize_circuit(self):
        """Creates variational circuit."""
        C = Circuit(1)
        for l in range(self.glayers):
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
        for i in range(0, 4 * self.glayers, 4):
            params.append(self.params[i] * x + self.params[i + 1]) # x is scalar in this case
            params.append(self.params[i + 2] * x + self.params[i + 3])  # x is scalar in this case
        self._circuit.set_parameters(params)
        return self._circuit
  
    # discriminator circuit      
    def set_dparameters(self, new_params):
        """Method for updating parameters of the class.
        Args:
            new_params (array): New parameters to update
        """
        self.dparams = new_params       
    
    def _initialize_dcircuit(self):
        """Creates variational circuit."""
        D = Circuit(1)
        for l in range(self.dlayers):
            D.add(gates.RY(0, theta=0))
            D.add(gates.RZ(0, theta=0))
        return D            

    def dcircuit(self, x):
        """Method creating the circuit for a point (in the datasets).
        Args:
            x (array): Point to create the circuit.
        Returns:
            Qibo circuit.
        """
        dparams = []
        for i in range(0, 4 * self.dlayers, 4):
            dparams.append(self.dparams[i] * x + self.dparams[i + 1]) # x is scalar in this case
            dparams.append(self.dparams[i + 2] * x + self.dparams[i + 3])  # x is scalar in this case
        self._dcircuit.set_parameters(dparams)
        return self._dcircuit
  
    def dpredict(self, xval):   
             
        # need a blank state to contract fidelity against
        blank_state = [np.array([1, 0], dtype='complex'), np.array([0, 1], dtype='complex')]
        
        labels = [[0]] * len(xval[0])
        for j, x in enumerate(xval[0]):
            D = self.dcircuit(x)
            state = D.execute()
            fids = np.empty(len(blank_state))
            for i, t in enumerate(blank_state):
                fids[i] = fidelity(state, t)
            labels[j] = float(np.argmax(fids))

        return labels   
  
    
    def set_seed(self, sseed):
        self.seed=sseed
                      
    def cost_function(self, params=None):
        
        # setup parameters
        if params is None:
            params = self.params       

        self.set_parameters(params)
               
        # First create another set of fake data, using the gparams, this time the labels are set to 1
        gseed=self.seed
        xinput = create_dataset(self.nmeas,1,gseed)
        xfake = self.generate(xinput,params)
        yfake = np.ones(self.nmeas)
        #print(xfake)
        
        # Now guess the labels using the discriminator
        yguess = self.dpredict([xfake])
        #print(yfake,yguess)
        
        
        cf = tf.keras.losses.binary_crossentropy(yfake, yguess)
        #cf = tf.reduce_mean(cf)
        
        cf /= self.nmeas
    
        #tflabel = tf.convert_to_tensor(cf, dtype=tf.float64)
        #cf=(tflabel)
        #print(params,float(cf))
        
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


    def generate(self, xval, params=None):
        
        if params is None:
            params = self.params       

        self.set_parameters(params)        
        
        
        y=np.zeros((len(xval[0])))
        
        i=0
        for x in xval[0]:
            
            C = self.circuit(x)
            state = C.execute()
            y[i] = qgen_real_out(state)
            i+=1
                
    
        return y  


def qgen_real_out(state):
    hx  = hamiltonians.X(1, numpy=True) # create a 1-qubit Pauli-X Hamiltonian
    num = hx.expectation(state).numpy().real # project the state (output from generator) with Hamiltonian 
    res = (1 - num) / (1+num) # relate this way, it's positive
    #res = np.abs(num) # make it easier for the network by mapping the result between 0 and 1 
    #res = num # between -1 and 1
    return res
         
def fidelity(state1, state2):
    return tf.constant(tf.abs(tf.reduce_sum(tf.math.conj(state2) * state1))**2)

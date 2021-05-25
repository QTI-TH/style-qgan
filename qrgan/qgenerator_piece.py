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
from datasets import create_dataset, create_target
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
print("Qibo runs with "+str(current_threads)+" thread(s)")


class single_qubit_generator:
    def __init__(self, layers, dlayers, grid=None, seed=0):
       
       
        print('# Reading qlassifier parameters into generator...')
        #qlassi = np.loadtxt("./out.qlassi.parameters");
        #self.qlassi = qlassi
        
        np.random.seed(seed)
        self.layers = layers
        self.dlayers = dlayers
        
        self.data_set = create_dataset()
        self.target = create_target('gauss')
               
        outf = open("./out.qgen.target", "w")
        for x in self.target:
             outf.write("%.7e\n" % ( x ))
        outf.close
        
        
        self.params = np.random.randn(layers * 4)
        self._circuit = self._initialize_circuit() # initialise with random parameters
        
        # some tests of the initial labels and cost function labels 
        print("# Testing, no minimisation:")
        test_out=self.cost_function()
        print("# --- initial cost function: {} ".format(test_out))
                    
        try:
            os.makedirs('results/generate'+self.name+'/%s_layers' % self.layers)
        except:
            pass
            

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
  
                       
    def cost_function(self, params=None):
        """Method for computing the cost function for the training set, using fidelity.
        Args:
            params(array): new parameters to update before computing
        Returns:
            float with the cost function.
        """
        #print(params)

        if params is None:
            params = self.params       

        self.set_parameters(params)        
        
        cf=0 
        
        #print(self.data_set.shape)         
        #print(len(self.data_set))
        #print(len(self.data_set[0]))
         
        #for i in range(0,len(self.data_set)):
        for i in range(0,len(self.data_set)):
            
            y=np.zeros(len(self.data_set[0])) 
            for j in range(0,len(self.data_set[0])):
                
                # generate the real output from our circuit
                x = self.data_set[i,j]       
                C = self.circuit(x)
                state1 = C.execute()
                y[j] = qgen_real_out(state1)
             
                    
            # "fake" discriminator, do the Kolmogorov-Smirnoff test
            kstst=ks_2samp(y,self.target)
            
            # yes/no approach: if the p-value of the KS test is larger than 0.4 accept the result 
            #if kstst[1] > 0.4:
            #    cf+=0
            #else:
            #    cf+=1    
            
            # score approach: return KS p-value as score to be minimised, i.e. p=1 -> cf+=0, p=0 -> cf+=1
            cf+=(1 - kstst[1])
                
                
            
            # Kullbeck-Leibler needs conversion from distributed points to pdf
            #kldv=entropy(self.target)
            #print(kldv)    
            
        #cf /= len(self.data_set)
        
        tflabel = tf.convert_to_tensor(cf, dtype=tf.float64)
        cf=(tflabel)
        #print(params,float(cf))
        
        return cf    
        
    

    def minimize(self, method='BFGS', options=None, compile=True):
        loss = self.cost_function
        print("# Run minimisation:")


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


    def generate(self):
        """Method for predicting data
        Returns:
            files with data etc
        """
        
        outf = open("./out.qgen.samples", "w")
        
        nsamples=1000        
        for i in range(0,nsamples):
            
            xwindow=1 # between -xwindow and xwindow
            x=float(xwindow * ( 1 - 2 * np.random.rand(1, 1)))

            C = self.circuit(x)
            state = C.execute()
            y = qgen_real_out(state)
                    
            outf.write("%.7e %.7e\n" % ( y, x ))
        
        outf.close
        
        
        return 0    


def qgen_real_out(state):
    hx  = hamiltonians.X(1, numpy=True) # create a 1-qubit Pauli-X Hamiltonian
    num = hx.expectation(state).numpy().real # project the state (output from generator) with Hamiltonian 
    #res = (1 - num) / (1+num) # relate this way, it's positive
    #res = np.abs(num) # make it easier for the network by mapping the result between 0 and 1 
    res = num # between -1 and 1
    return res
         
def fidelity(state1, state2):
    return tf.constant(tf.abs(tf.reduce_sum(tf.math.conj(state2) * state1))**2)

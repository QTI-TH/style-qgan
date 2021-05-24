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
print("Qibo runs with "+str(current_threads)+" thread(s)")


class single_qubit_classifier:
    def __init__(self, layers, grid=None, seed=0):
       
       
        print('# Reading qlassifier parameters into generator...')
        #qlassi = np.loadtxt("./out.qlassi.parameters");
        #self.qlassi = qlassi
        
        np.random.seed(seed)
        self.layers = layers
        
        self.training = create_training('gauss')
               
        #outf = open("./out.qdsc.training", "w")
        #for x in self.training:
        #     outf.write("%.7e\n" % ( x ))
        #outf.close
        
        
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
        
        # need a blank state to contract fidelity against
        blank_state = [np.array([1, 0], dtype='complex'), np.array([0, 1], dtype='complex')]
        
        # setup parameters
        if params is None:
            params = self.params       

        self.set_parameters(params)        
        
        #print(self.training[0], self.training[1])
        
        # initiate cost function and calculate it
        cf=0
        tots=len(self.training[0])
        quali=0
        
        for i in range(0,len(self.training[0])):
                    
            slabl=0    
            for x in self.training[0][i]:
                #y=self.training[1][i]
                #print(x,y)   
            
                # generate the output from our circuit     
                C = self.circuit(x)
                state1 = C.execute()   
                 
                # label is projected out via fidelity: y=yes -> blank_state[1], y=no -> blank_state[0]
                #slabl += .5 * (1 - fidelity(state1, blank_state[int(y)])) ** 2
                # not a good cost function, would prefer something that is based on how many it got right. Like in the predict case
           
           #slabl/=(len(self.training[0][0]))        
           #cf+=slabl
           
                # from predict, in this cost function cf=0 when all labels are right
                fids = np.empty(len(blank_state))
                for j, t in enumerate(blank_state):
                    fids[j] = fidelity(state1, t)
                slabl += np.argmax(fids)
                
            slabl/=(len(self.training[0][0]))     

            qtest = self.training[1][i] - slabl
            if qtest == 0.0:
                quali+=1
                
        cf=tots-quali    
        
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



    def predict(self):
    
  
        # need a blank state to contract fidelity against
        blank_state = [np.array([1, 0], dtype='complex'), np.array([0, 1], dtype='complex')]
        
        tots=len(self.training[0])
        quali=0
        for i in range(0,len(self.training[0])):
            
            slabl=0
            for x in self.training[0][i]:
                
                # generate the output from our circuit     
                C = self.circuit(x)
                state1 = C.execute()

                fids = np.empty(len(blank_state))
                for j, t in enumerate(blank_state):
                    fids[j] = fidelity(state1, t)
                slabl += np.argmax(fids)
                
            slabl/=(len(self.training[0][0]))     

            qtest = self.training[1][i] - slabl
            if qtest == 0.0:
                quali+=1
            
        print("# The discriminator got {} of {} right in training".format(quali, tots))    
            
        return quali, tots


def qgen_real_out(state):
    hx  = hamiltonians.X(1, numpy=True) # create a 1-qubit Pauli-X Hamiltonian
    num = hx.expectation(state).numpy().real # project the state (output from generator) with Hamiltonian 
    #res = (1 - num) / (1+num) # relate this way, it's positive
    #res = np.abs(num) # make it easier for the network by mapping the result between 0 and 1 
    res = num # between -1 and 1
    return res
         
def fidelity(state1, state2):
    return tf.constant(tf.abs(tf.reduce_sum(tf.math.conj(state2) * state1))**2)

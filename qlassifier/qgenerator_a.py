from qibo.models import Circuit
from qibo import hamiltonians, gates, models
import numpy as np
from datasets_a import create_dataset, create_target
import tensorflow as tf
import os


class single_qubit_generator:
    def __init__(self, name, layers, dlayers, grid=None, test_samples=200, seed=0):
       
       
        print('# Reading qlassifier parameters into generator...')
        qlassi = np.loadtxt("./out.qlassi.parameters");
        self.qlassi = qlassi
        
        np.random.seed(seed)
        self.name = name
        self.layers = layers
        self.dlayers = dlayers
        
        self.data_set = create_dataset(name, grid=grid, samples=test_samples)
        self.target = create_target(name)
        
        self.params = np.random.randn(layers * 4)
        self._circuit = self._initialize_circuit() # initialise with random parameters
        self._dcircuit = self._initialize_dcircuit() # initialise the discriminator circuit
        
        # some tests of the initial labels and cost function labels 
        print("# Testing labels, no minimisation:")
        test_in=np.sum(self.eval_test_set_fidelity())/test_samples
        test_out=self.cost_function_fidelity()
        print("# --- Cost from initial set: {} and randomly transformed set {}".format(test_in,test_out))
        
#        np.random.seed(seed*2)
#        self.params = np.random.randn(layers * 4)
#        test_in=np.sum(self.eval_test_set_fidelity())/test_samples
#        test_out=self.cost_function_fidelity()
#        print("# --- Cost from initial set: {} and randomly transformed set {}".format(test_in,test_out))      

#        np.random.seed(seed*20)
#        self.params = np.random.randn(layers * 4)
#        test_in=np.sum(self.eval_test_set_fidelity())/test_samples
#        test_out=self.cost_function_fidelity()
#        print("# --- Cost from initial set: {} and randomly transformed set {}".format(test_in,test_out))      
        
#        print("# --- Note: The initial set also provides the first guess for the minimiser,") 
#        print("# --- if it's already randomly good say cf~=0.1, this makes the job easier.")
        
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

    # discriminator circuit, must be same as before!
    def _initialize_dcircuit(self):
        """Creates variational circuit."""
        C = Circuit(1)
        for l in range(self.dlayers):
            C.add(gates.RY(0, theta=0))
            C.add(gates.RZ(0, theta=0))
        return C
        
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
            params.append(self.params[i] * x[0] + self.params[i + 1])
            params.append(self.params[i + 2] * x[1] + self.params[i + 3])
        self._circuit.set_parameters(params)
        return self._circuit
   
    def dcircuit(self, x):
        """Method creating the circuit for a point (in the datasets).
        Args:
           x (array): Point to create the circuit.
        Returns:
           Qibo circuit for predicting the discriminator result.
        """
        qlassi = []
        for i in range(0, 4 * self.dlayers, 4):
           qlassi.append(self.qlassi[i] * x[0] + self.qlassi[i + 1])
           qlassi.append(self.qlassi[i + 2] * x[1] + self.qlassi[i + 3])
        self._dcircuit.set_parameters(qlassi)
        return self._dcircuit


    def cost_function_one_point_fidelity(self, x):
        """Method for computing the cost function for
        a given sample (in the datasets), using fidelity.
        Args:
            x (array): Point to create the circuit.
            y (int): label of x.
        Returns:
            float with the cost function.
        """
        
        # generate the real output from our circuit
        C = self.circuit(x)
        state1 = C.execute()
        y = qgen_real_out(state1)
           
        
        # generate the labels based on the discriminator circuit
        x2 = ([x[0],y]) # join x and y_new values
        #print(x[0],x[1],y,x2,x)
        
        D = self.dcircuit(x2) 
        state2 = D.execute()
        fids = np.empty(len(self.target))
        for i, t in enumerate(self.target):
            fids[i] = fidelity(state2, t)
        label = np.argmax(fids)
        
        # next define cost from this point. 0 if the label is 1 (right) or 1 if the label is 0 (wrong)
        
        #print(x,y,label,cf)
        tflabel = tf.convert_to_tensor(label, dtype=tf.float64)
        cf=(1.-tflabel)
        
        return cf
                    

    def cost_function_fidelity(self, params=None):
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
                  
        cf = 0
        for x in self.data_set[0]:
            cf += self.cost_function_one_point_fidelity(x) 
            #print(cf)    
        cf /= len(self.data_set[0])
        
        #print(params,float(cf))
        return cf    
        
        
    def eval_test_set_fidelity(self): # prediction circuit here, just for testing
        """Method for evaluating points in the training set, using fidelity.
        Returns:
            list of guesses for the discriminator parameters.
        """
        labels = [[0]] * len(self.data_set[0])
        for j, x in enumerate(self.data_set[0]):
            C = self.circuit(x) 
            state = C.execute()
            fids = np.empty(len(self.target))
            for i, t in enumerate(self.target):
                fids[i] = fidelity(state, t)
            labels[j] = np.argmax(fids)
            
            # test the state projection here
            # num = qgen_real_out(state)
            # print(x,num)

        return labels        


    def minimize(self, method='BFGS', options=None, compile=True):
        loss = self.cost_function_fidelity
        print("# Run minimisation:")


        if method == 'cma':
            # Genetic optimizer
            import cma
            #r = cma.fmin2(lambda p: loss(p).numpy(), self.params, 2)
            r = cma.fmin2(lambda p: loss(p).numpy(), self.params, 2, {'seed':113895})
            result = r[1].result.fbest
            parameters = r[1].result.xbest

        elif method == 'sgd':
            from qibo.tensorflow.gates import TensorflowGate
            circuit = self.circuit(self.training_set[0])
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
        self.generate_set = create_dataset("uniform", grid=None, samples=nsamples)
                
        for x in self.generate_set[0]:
            C = self.circuit(x)
            state = C.execute()
            y = qgen_real_out(state)
                    
            outf.write("%.7e %.7e\n" % ( x[0],y ))
        
        outf.close
        
        
        return 0    


def qgen_real_out(state):
    hx  = hamiltonians.X(1, numpy=True) # create a 1-qubit Pauli-X Hamiltonian
    num = hx.expectation(state).numpy().real # project the state (output from generator) with Hamiltonian 
    #res = (1 - num) / (1+num) # relate this way
    res = np.abs(num) # make it easier for the network by mapping the result between 0 and 1 
    return res
         
def fidelity(state1, state2):
    return tf.constant(tf.abs(tf.reduce_sum(tf.math.conj(state2) * state1))**2)

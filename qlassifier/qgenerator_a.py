from qibo.models import Circuit
from qibo import gates
import numpy as np
from datasets_a import create_dataset, create_target
import tensorflow as tf
import os


class single_qubit_generator:
    def __init__(self, name, layers, grid=None, test_samples=100, seed=1):
       
       
        print('# Reading qlassifier parameters into generator...')
        qlassi = np.loadtxt("./out.qlassi.parameters");
        self.qlassi = qlassi
        
        np.random.seed(seed)
        self.name = name
        self.layers = layers
        
        self.data_set = create_dataset(name, grid=grid, samples=test_samples)
        self.target = create_target(name)
        
        self.params = np.random.randn(layers * 4)
        self._circuit = self._initialize_circuit() # initialise with random parameters
        
        print(self.eval_test_set_fidelity())
        print(self.cost_function_fidelity())
        
        np.random.seed(seed*2)
        self.params = np.random.randn(layers * 4)
        print(self.eval_test_set_fidelity())
        print(self.cost_function_fidelity())
        
        
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
        for i in range(0, 4 * self.layers, 4):
           qlassi.append(self.qlassi[i] * x[0] + self.qlassi[i + 1])
           qlassi.append(self.qlassi[i + 2] * x[1] + self.qlassi[i + 3])
        self._circuit.set_parameters(qlassi)
        return self._circuit


    def cost_function_one_point_fidelity(self, x, y):
        """Method for computing the cost function for
        a given sample (in the datasets), using fidelity.
        Args:
            x (array): Point to create the circuit.
            y (int): label of x.
        Returns:
            float with the cost function.
        """
        C = self.circuit(x)
        state = C.execute()
        cf = 0
        return cf


    def cost_function_fidelity(self, params=None):
        """Method for computing the cost function for the training set, using fidelity.
        Args:
            params(array): new parameters to update before computing
        Returns:
            float with the cost function.
        """
        if params is None:
            params = self.params

        self.set_parameters(params)
        
        tag   = self.eval_test_set_fidelity()
        val   = self.data_set[1]
        xy    = self.data_set[0]
        x, y  = xy[:, 0], xy[:, 1]
        #print(x,y,tag,val)
        
       
            
        cf = 0
        return cf
        
    def eval_test_set_fidelity(self):
        """Method for evaluating points in the training set, using fidelity.
        Returns:
            list of guesses for the discriminator parameters.
        """
        labels = [[0]] * len(self.data_set[0])
        for j, x in enumerate(self.data_set[0]):
            C = self.dcircuit(x) # set the discriminator circuit here
            state = C.execute()
            fids = np.empty(len(self.target))
            for i, t in enumerate(self.target):
                fids[i] = fidelity(state, t)
            labels[j] = np.argmax(fids)

        return labels        


    def minimize(self, method='BFGS', options=None, compile=True):
        loss = self.cost_function_fidelity

        import numpy as np
        from scipy.optimize import minimize
        m = minimize(lambda p: loss(p).numpy(), self.params,
                     method=method, options=options)
        result = m.fun
        parameters = m.x

        return result, parameters



        
def fidelity(state1, state2):
    return tf.constant(tf.abs(tf.reduce_sum(tf.math.conj(state2) * state1))**2)

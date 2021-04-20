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
        
        self.data_set = create_dataset(name, grid=grid)
        self.target = create_target(name)
        
        self.params = np.random.randn(layers * 4)
        self._circuit = self._initialize_circuit() # initialise with random parameters
        
        
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
       return self._dcircuit
        
        

        
def fidelity(state1, state2):
    return tf.constant(tf.abs(tf.reduce_sum(tf.math.conj(state2) * state1))**2)

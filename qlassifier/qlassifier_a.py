from qibo.models import Circuit
from qibo import gates
import numpy as np
from datasets_a import create_dataset, create_target
import tensorflow as tf
import os


class single_qubit_classifier:
    def __init__(self, name, layers, grid=None, test_samples=100, seed=0):
        """Class with all computations needed for classification.
        Args:
            name (str): Name of the problem to create the dataset, to choose between
                ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines'].
            layers (int): Number of layers to use in the classifier.
            grid (int): Number of points in one direction defining the grid of points.
                If not specified, the dataset does not follow a regular grid.
            samples (int): Number of points in the set, randomly located.
                This argument is ignored if grid is specified.
            seed (int): Random seed.
        Returns:
            Dataset for the given problem (x, y).
        """
        np.random.seed(seed)
        self.name = name
        self.layers = layers
        self.training_set = create_dataset(name, grid=grid)
        
        
        
        self.test_set = create_dataset(name, samples=test_samples)
        self.target = create_target(name)
        self.params = np.random.randn(layers * 4)
        self._circuit = self._initialize_circuit()
        
        try:
            os.makedirs('results/'+self.name+'/%s_layers' % self.layers)
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
        cf = .5 * (1 - fidelity(state, self.target[y])) ** 2
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
        cf = 0
        for x, y in zip(self.training_set[0], self.training_set[1]):
            cf += self.cost_function_one_point_fidelity(x, y)
        cf /= len(self.training_set[0])
        
        #print(params,float(cf))
        return cf

    def minimize(self, method='BFGS', options=None, compile=True):
        loss = self.cost_function_fidelity

        if method == 'cma':
            # Genetic optimizer
            import cma
            r = cma.fmin2(lambda p: loss(p).numpy(), self.params, 2)
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
            m = minimize(lambda p: loss(p).numpy(), self.params,
                         method=method, options=options)
            result = m.fun
            parameters = m.x

        return result, parameters

    def eval_test_set_fidelity(self):
        """Method for evaluating points in the training set, using fidelity.
        Returns:
            list of guesses.
        """
        labels = [[0]] * len(self.test_set[0])
        for j, x in enumerate(self.test_set[0]):
            C = self.circuit(x)
            state = C.execute()
            fids = np.empty(len(self.target))
            for i, t in enumerate(self.target):
                fids[i] = fidelity(state, t)
            labels[j] = np.argmax(fids)

        return labels


    def output(self):
        """Method for outputting data
        Returns:
            files with data etc
        """
        
        outf0  = open("./out.qlassi.data.label0", "w")
        outf1  = open("./out.qlassi.data.label1", "w")
        
        xy    = self.training_set[0]
        x, y  = xy[:, 0], xy[:, 1]
        labels = self.training_set[1]
        
        
        for n in range(0,len(labels)):
            if labels[n]==0:
                outf0.write("%.7e %.7e\n" %(x[n],y[n]))
            else:
                outf1.write("%.7e %.7e\n" %(x[n],y[n]))
        
        
        outf0.close
        outf1.close
        
        return 0
        
        
    def predict(self):
        """Method for predicting data
        Returns:
            files with data etc
        """
        
        outf3  = open("./out.qlassi.predict.label0", "w")
        outf4  = open("./out.qlassi.predict.label1", "w")
        
        xy    = self.test_set[0]
        x, y  = xy[:, 0], xy[:, 1]
        labels = self.eval_test_set_fidelity()
    
        #print(labels)
        
        for n in range(0,len(labels)):
            if labels[n]==0:
                outf3.write("%.7e %.7e\n" %(x[n],y[n]))
            else:
                outf4.write("%.7e %.7e\n" %(x[n],y[n]))
        
        outf3.close
        outf4.close
        
        return 0    

    

def fidelity(state1, state2):
    return tf.constant(tf.abs(tf.reduce_sum(tf.math.conj(state2) * state1))**2)

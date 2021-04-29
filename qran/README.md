# qgmc - qran (quantum regressive adversarial network)


- **Discriminator (D):** 
  - Single qubit 
  - re-uploading is used to train a classifier on data to distinguish elements of a wanted result function

- **Generator (G):** 
  - single qubit 
  - task is to transform input=(x,y) to output=(x,y_new), where y_new is part of the target function. 
  - the discriminator evaluates the result by asigning labels to the y_new. The cost function is minimised when all labels are 1=in the distribution


# Code info

Main: qran.py
Requires: dataset.py, qlassifier.py, qgenerator.py

Routines to generate training data sets. Currently implemented (dataset.py):
-   create_dataset: Create uniformly distributed random points in x=(x_1,x_2) and call subroutine to add label (y), output is tuple (x,y)
-   create_target: Creates target array of complex 1+i0 and 0+i1
-   uniform: Subroutine that creates uniformly distributed random real labels, needed for qgenerator
-   gauss: Subroutine to imprint labels on training set according to a given Gaussian distribution.
-   gauss2: Subroutine to imprint labels on training set according to a given Gaussian distribution, where a fixed percentage is inside  the Gaussian tube

Routines to run the QML quantum discriminator (qlassifier.py)
- single_qubit_qlassifier: Defines and initiates qlassifier
-   set_parameters: set parameters in quantum circuit
-   _initialise_circuit: initalised generator circuit
-   circuit: define the circuit
-   cost_function_one_point_fidelity: Return the cost of input point, executes circuit for label and evaluates cost based on label.
-   cost_function_fidelity: gather cost for all points
-   minimize: run the minimizer, currently cma and scipy.minimize
-   eval_test_set_fidelity: run discriminator circuit and generate label after training
-   output: print training data set, separated by label 0 and 1
-   predict: predict labels on test_set and output
-   fidelity: multiply circuit state and target state (from create_target) into result 0 or 1

Routines to run the QML quantum generator (qgenerator.py): 
-   single_qubit_generator: Defines and initiates qgenerator
-   set_parameters: set parameters in quantum circuit
-   _initialise_circuit: initalised generator circuit
-   _initialise_dcircuit: initalised discriminator circuit
-   circuit: define the circuit
-   cost_function_one_point_fidelity: Return the cost of input point, executes generator circuit to create data, then the discriminator circuit for label and evaluates cost based on label.
-   cost_function_fidelity: gather cost for all points
-   minimize: run the minimizer, currently cma and scipy.minimize
-   generate: generate data set by executing generator circuit given some input points
-   qgen_real_out: project curcuit result agains Pauli_X Hamiltonian to output real numbers
-   fidelity: multiply circuit state and target state (from create_target) into result 0 or 1
    


# Example 

**Full network**
Discriminator quality (orange): n_layers=2, n_input=20000, cma algorithm
<img width="649" alt="qd" src="https://github.com/scarrazza/qgmc/files/6398030/qlassifier.pdf>

Generator data (orange): n_layers=2, n_generate=2000, cma algorithm 
<img width="649" alt="qg" src="https:https://github.com/scarrazza/qgmc/files/6398035/qgenerator.pdf>



# Re-uploading

- [Re-uploading paper](https://arxiv.org/abs/1907.02085)

- [Re-uploading example in qibo](https://qibo.readthedocs.io/en/stable/tutorials/reuploading_classifier/README.html)


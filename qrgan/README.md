# qgmc - qrgan (quantum re-uploading generative adversarial network)


- **Discriminator (D):** 
  - Single qubit 
  - re-uploading is used to train a classifier on data to distinguish elements of a wanted result function
  - currently not implemented
  	- instead: Kolmogorov-Smirnov test is performed comparing generator data and target data, if p-value>0.4 the label "correct" is returned.

- **Generator (G):** 
  - single qubit 
  - task is to transform input=(x) to output=(x_new), where x_new is distributed as the target distribution. 
  - the discriminator evaluates the result by asigning labels to the y_new. The cost function is minimised when all labels are 1=in the distribution

** Implementation **

Generator recipe:
- generate target sample by pulling n numbers from a Gaussian distribution (python implementation)
- generate uniform noise dataset, its shape is (n_samp,n_meas), where n_samp gives the numer of "samples" that are put through the KS test and n_meas is the number of points per sample
- initiate quantum circuit parameters randomly then:
  - draw parameters (skipped at first step)
  - execute the circuit once for all n_meas
  - perform KS test, cost=0 for correct and 1 for wrong
  - iterate over all n_samp and collect results for cost in cf
  - return cf for minimisation
  - repeat until minimised
  

# Code info

Main: qrgan.py

Requires: dataset.py, qgenerator.py

Routines to generate training data sets. Currently implemented (dataset.py):
-   create_dataset: Create uniformly distributed random points in x=(x_1,x_2) and call subroutine to add label (y), output is tuple (x,y)
-   create_target: Creates target array of complex 1+i0 and 0+i1

Routines to run the QML quantum generator (qgenerator.py): 
-   single_qubit_generator: Defines and initiates qgenerator
-   set_parameters: set parameters in quantum circuit
-   _initialise_circuit: initalised generator circuit
-   circuit: define the circuit
-   cost_function: over one set of measurements = sample perform KS test and return 1 or 0, iterating over all samples gives the cost function to minimise
-   minimize: run the minimizer, currently cma and scipy.minimize
-   generate: generate data set by executing generator circuit given some input points
-   qgen_real_out: project curcuit result agains Pauli_X Hamiltonian to output real numbers
-   fidelity: multiply circuit state and target state (from create_target) into result 0 or 1
    


# Example 

**Generator**

- Generator data (orange): n_layers=4, n_target=2000, n_samples=200, n_meas=10, n_generate=10000, cma algorithm (maxiter=100)

<img width="649" alt="qg" src="https://user-images.githubusercontent.com/11166117/117848299-777be400-b283-11eb-8423-0bd54ff46007.png">


- Generator data (orange): n_layers=2, n_target=2000, n_samples=200, n_meas=10, n_generate=10000, cma algorithm (maxiter=100)

<img width="649" alt="qg" src="https://user-images.githubusercontent.com/11166117/117848504-aa25dc80-b283-11eb-95c0-bf50fa41c9e9.png">



# Re-uploading

- [Re-uploading paper](https://arxiv.org/abs/1907.02085)

- [Re-uploading example in qibo](https://qibo.readthedocs.io/en/stable/tutorials/reuploading_classifier/README.html)


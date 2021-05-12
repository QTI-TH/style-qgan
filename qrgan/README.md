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

## Implementation 

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

Routines to generate data sets. Currently implemented (dataset.py):
-   create_dataset: Create uniformly distributed points in (n_samp,n_meas), **redundant once we inject quantum random numbers**
-   create_target: Creates data points x distributed according to the target distribution, options: gauss or lognormal

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

## Some observations

The minimisation does not go very well. In these examples the worst case for the two distributions D_target != D_generated implies a cf=200 (i.e. cost=1 for each of the test samples). Typically the cf reached at the iteration cutoff=100 is around cf=100-120. This implies that of 200 samples generated only 80-100 will actually lie in the target distribution on average. In that light it makes sense that all figures exhibit a strong uniformly distributed background. It is likely that replacing the simple yes/no-based cost function with one in which the KS distance is minimised instead will work better. However, once a discriminator comes into play that won't work anymore since it will reply the yes/no answers as is done here. The idea was indeed to mimic the discriminator response in this way.

## Generator, Gaussian, yes/no criterion

- Generator setup: n_layers=4, n_target=2000, n_samples=200, n_meas=10, n_generate=10000, cma algorithm (maxiter=100)

<img width="649" alt="qg" src="https://user-images.githubusercontent.com/11166117/117848299-777be400-b283-11eb-8423-0bd54ff46007.png">


- Generator setup: n_layers=2, n_target=2000, n_samples=200, n_meas=10, n_generate=10000, cma algorithm (maxiter=100)

<img width="649" alt="qg" src="https://user-images.githubusercontent.com/11166117/117848504-aa25dc80-b283-11eb-95c0-bf50fa41c9e9.png">

## Generator, Lognormal, yes/no criterion

- Generator setup: n_layers=2, n_target=2000, n_samples=200, n_meas=10, n_generate=10000, cma algorithm (maxiter=100)

<img width="649" alt="qg" src="https://user-images.githubusercontent.com/11166117/117863900-8919b780-b294-11eb-9b49-edb3645aa884.png">

## Generator, Lognormal, score criterion

- Generator setup: n_layers=2, n_target=2000, n_samples=200, n_meas=10, n_generate=10000, cma algorithm (maxiter=100)
- 
<img width="649" alt="qg" src="https://user-images.githubusercontent.com/11166117/117973536-d3e70e00-b32c-11eb-916b-408af05ec2b2.png">



# Re-uploading

- [Re-uploading paper](https://arxiv.org/abs/1907.02085)

- [Re-uploading example in qibo](https://qibo.readthedocs.io/en/stable/tutorials/reuploading_classifier/README.html)


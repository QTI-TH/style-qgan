# qgmc
Quantum Generative MC


- We want to research a quantum based machine learning method for applications in generative Monte-Carlo, in particular sampling of event distributions.
- Currently we believe a quantum GAN approach is the most promising. For the alternative, a quantum variational autoencoding approach, it is not clear to us how to implement the sampling on the latent space at the moment. 
- We do not want to encode on the qubit angles directly so that we can avoid having to prepare |in> states. This makes the current PennyLane implementation less attractive but an approach using re-uploading more so.
- Redudant qubits could be used in two ways:
  - Implement error mitigation for more robust results.
  - Encode extra information or constraints.


## Planned network structure

We have rough ideas for the GAN structure

- **Discriminator (D):**
  - single qubit "wire" to output 0 or 1. (Insert graphical representation here)

- **Generator (G):**
  - one qubit per observable (aim: s,t,y) -> three wires
  - introduction of uniformly distributed random variables, either via classical RNG (not preferred) or quantum RNG (preferred) (Insert graphical representation here)

- **quantum GAN:**
  - Do we need a 4th wire to connect the Generator and Discriminator? 
  - In D, could we use quantum interference to project (like e.g. in Grover's algorithm) instead of a classical minimizer? Could the whole problem not be recast in such a way? In that case we could use quantum parallelism more efficiently?

# Ressources

## References

- [QuGAN](https://arxiv.org/pdf/2010.09036.pdf)
- [QGAN](https://www.nature.com/articles/s41534-019-0223-2.pdf)
- [QVA](https://arxiv.org/pdf/2010.06599.pdf)

## Running, improving and debugging a classical machine learning setup

- [Ray tune](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
- [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb), a space where we can write and run on GPU/TCPs for free should more "umpf" than a laptop be required. It can also be hooked up to our github.

## GAN implementations

- [Classical GAN](https://github.com/luigifvr/tesi/tree/master/dcgan)

- [PernnyLane GAN](https://pennylane.ai/qml/demos/tutorial_QGAN.html)

- [qiskit GAN](https://github.com/keamanansiber/qiskit/blob/master/3QuantumMachineLearning/qGAN_LoadingRandomDistributions.ipynb)

## Re-uploading

- [Re-uploading](https://arxiv.org/abs/1907.02085)

- [Implementation in PennyLane](https://pennylane.ai/qml/demos/tutorial_data_reuploading_classifier.html)

- [PDF paper by Stefano et al.](https://arxiv.org/abs/2011.13934)

- [PDF implementation in qibo](https://qibo.readthedocs.io/en/stable/tutorials/qPDF/qPDF.html)


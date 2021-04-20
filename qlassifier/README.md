# qgmc - qlassifier

- **Discriminator (D) development:** 
  - single qubit "wire" to output 0 or 1. 
  - use re-uploading for this task

The initial upload is the tutorial implementation of the re-uploading classifier in qibo. Based on this we should be able to learn how to classify also other scenarios and to write our own code to perform the task.


# Code info

- **_a.py: First implementation aiming to classify a Gaussian.**
  - simplified the tutorial code
  - **create_dataset** with grid=None creates a set of points with x=[-xwindow,xwindow] and y=[0,1]
  - **create_target(gauss)** will read the random points and if one of them is within cutoff of a Gaussian distribution with parameters (m,sig) will set the label to 1 and 0 otherwise.
  - **create_target(gauss2)** creates a more balanced training set: nratio sets the ratio of 0 vs. 1 labels in the training, setting it e.g. to 0.5 means the data is arranged in such a way that half the samples have label 0 will the remainder is within a Gaussian tube defined by (m,sig,cutoff).
  - running on a laptop is fairly slow, tests with 4 layers already take a long time. It'd be good to run this on a gpu or so. I couldn't port it to google colab easily though. Does anybody have good experience porting simple .py to ipynb?
   
# Some results 

Gaussian parameters: m=0, sig=0.5, cutoff=0.1, n_samples=200, n_predict=100, nlayers=4
<img width="649" alt="our-gan" src="https://github.com/scarrazza/qgmc/files/6342777/qlassifier.nl4.ns200.nr05.pdf">

Gaussian parameters: m=0, sig=0.5, cutoff=0.1, n_samples=400, n_predict=100, nlayers=4
<img width="649" alt="our-gan" src="https://github.com/scarrazza/qgmc/files/6342776/qlassifier.nl4.ns400.nr05.pdf">


# Ressources

## Re-uploading

- [Re-uploading paper](https://arxiv.org/abs/1907.02085)

- [Re-uploading example in qibo](https://qibo.readthedocs.io/en/stable/tutorials/reuploading_classifier/README.html)


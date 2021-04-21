# qgmc - qlassifier

- **Discriminator (D) development:** 
  - single qubit "wire" to output 0 or 1. 
  - use re-uploading for this task

- **Generator (G) development:** 
  - single qubit "wire", 
  - task is to transform input=(x,y) to output=(x,y_new), where y_new is in the target distribution. 
  - the discriminator evaluates the result by asigning labels to the y_new. The cost function is minimised when all labels are 1=in the distribution


The initial upload is the tutorial implementation of the re-uploading classifier in qibo. Based on this we should be able to learn how to classify also other scenarios and to write our own code to perform the task.


# Code info

- **_a.py: First implementation aiming to classify and sample in a Gaussian.**
  - simplified the tutorial code
  - **create_dataset** with grid=None creates a set of points with x=[-xwindow,xwindow] and y=[0,1]
  - **create_target(gauss)** will read the random points and if one of them is within cutoff of a Gaussian distribution with parameters (m,sig) will set the label to 1 and 0 otherwise.
  - **create_target(gauss2)** creates a more balanced training set: nratio sets the ratio of 0 vs. 1 labels in the training, setting it e.g. to 0.5 means the data is arranged in such a way that half the samples have label 0 will the remainder is within a Gaussian tube defined by (m,sig,cutoff).
  - running on a laptop is fairly slow, tests with 4 layers already take a long time. It'd be good to run this on a gpu or so. I couldn't port it to google colab easily though. Does anybody have good experience porting simple .py to ipynb?
   
   
# Some results for the discriminator/generator combination

Full network: Generator data (orange), Discriminator quality (blue), n_layers=2, n_samples=100, cma algorithm

<img width="649" alt="our-gan" src="https://github.com/scarrazza/qgmc/files/6351036/qgenerator.nl2.ns100.pdf">

n_layers=2, n_samples=400, cma algorithm

<img width="649" alt="our-gan" src="https://github.com/scarrazza/qgmc/files/6351177/qgenerator.nl2.ns400.pdf">


Discriminator alone: n_layers=2, n_samples=200, scipy minimise algorithm

<img width="649" alt="our-gan" src="https://github.com/scarrazza/qgmc/files/6351035/qlassifier.nl2.ns200.nr05.cut01.pdf">
   
   
# Some results for the discriminator

- Gaussian parameters: m=0, sig=0.5, cutoff=0.1, n_samples=200, n_predict=100, nlayers=4
<img width="649" alt="our-gan" src="https://github.com/scarrazza/qgmc/files/6342777/qlassifier.nl4.ns200.nr05.pdf">

- Gaussian parameters: m=0, sig=0.5, cutoff=0.05, n_samples=200, n_predict=100, nlayers=4
<img width="649" alt="our-gan" src="https://github.com/scarrazza/qgmc/files/6342876/qlassifier.nl4.ns200.nr05.cut005.pdf">

- Gaussian parameters: m=0, sig=0.5, cutoff=0.05, n_samples=200, n_predict=100, nlayers=5
<img width="649" alt="our-gan" src="https://github.com/scarrazza/qgmc/files/6343303/qlassifier.nl5.ns200.nr05.cut005.pdf">

- Gaussian parameters: m=0, sig=0.5, cutoff=0.1, n_samples=400, n_predict=100, nlayers=4
<img width="649" alt="our-gan" src="https://github.com/scarrazza/qgmc/files/6342776/qlassifier.nl4.ns400.nr05.pdf">

- Gaussian parameters: m=0, sig=0.5, cutoff=0.1, n_samples=400, n_predict=100, nlayers=5
<img width="649" alt="our-gan" src="https://github.com/scarrazza/qgmc/files/6343458/qlassifier.nl5.ns400.nr05.cut005.pdf">


# Ressources

## Re-uploading

- [Re-uploading paper](https://arxiv.org/abs/1907.02085)

- [Re-uploading example in qibo](https://qibo.readthedocs.io/en/stable/tutorials/reuploading_classifier/README.html)


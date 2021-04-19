# qgmc 

- **Discriminator (D) development:** 
  - single qubit "wire" to output 0 or 1. 
  - use re-uploading for this task

The initial upload is the tutorial implementation of the re-uploading classifier in qibo. Based on this we should be able to learn how to classify also other scenarios and to write our own code to perform the task.

To-do list:
- Simplify the tutorial by removing unused options in "mod.py" (which is a copy of main)
- Reduce to one dataset
- Implement new dataset, ideally a simplified/idealised version of our even data
- Train and run qlassifier

# Code info

- **_a.py: First implementation aiming to classify a Gaussian.**
  - simplified the tutorial code
  - **create_dataset** with grid=None creates a set of points with x=[-1,1] and y=[0,1]
  - **create_target(gauss)** will read the random points and if one of them is within cutoff of a Gaussian distrition with parameters (m,sig) will set the label to 1 and 0 otherwise.
  - executing the code as usual runs and I get a cost function of 0.016 on a single layer and (m=0,sig=1,cutoff=0.02), but nothing is has been visually inspected for correctness.  

# Ressources

## Re-uploading

- [Re-uploading paper](https://arxiv.org/abs/1907.02085)

- [Re-uploading example in qibo](https://qibo.readthedocs.io/en/stable/tutorials/reuploading_classifier/README.html)


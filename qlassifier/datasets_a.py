import numpy as np
from itertools import product


def create_dataset(name, grid=None, samples=200, seed=0):
    """Function to create training and test sets for classifying.
    Args:
        name (str): Name of the problem to create the dataset, to choose between
            ['gauss', 'circle'].
        grid (int): Number of points in one direction defining the grid of points.
            If not specified, the dataset does not follow a regular grid.
        samples (int): Number of points in the set, randomly located.
            This argument is ignored if grid is specified.
        seed (int): Random seed
    Returns:
        Dataset for the given problem (x, y)
    """
    if grid == None:
        np.random.seed(seed)
        xwindow=1
        x = xwindow * ( 1 - 2 * np.random.rand(samples, 1)) # between -xwindow and xwindow
        y = np.random.rand(samples, 1) # between 0 and 1
        points = np.concatenate((x, y), axis=1)
    
    else:
        x = np.linspace(-1, 1, grid)
        points = np.array(list(product(x, x)))
    creator = globals()[f"_{name}"]

    x, y = creator(points)
    return x, y


def create_target(name):
    """Function to create target states for classification.
    Args:
        name (str): Name of the problem to create the target states, to choose between
            ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines']
    Returns:
        List of numpy arrays encoding target states that depend only on the number of classes of the given problem
    """
    if name in ['circle']:
        targets = [np.array([1, 0], dtype='complex'),
                   np.array([0, 1], dtype='complex')]
    elif name in ['gauss', 'gauss2']:
        targets = [np.array([1, 0], dtype='complex'),
                   np.array([0, 1], dtype='complex')]

    return targets



def _circle(points):
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(np.linalg.norm(points, axis=1) > np.sqrt(2 / np.pi))
    labels[ids] = 1

    return points, labels
    

# points from the randomly chosen grid withing distance cutoff of the target Gaussian are flagges with label 1. 
# Here there is naturally an imbalance between 0 and 1 labels    
def _gauss(points):
      
    def gaussian(x,m,sig):
        norm = 0.9 # puts whole range into random grid ranges, should be this: 1/(sig * np.sqrt(2*np.pi)) but we want it between 0 and 1
        res = norm * np.exp(-0.5 * ((x-m)/sig)**2 )
        return res
    
    def target_marker(x,m,sig,y,cutoff):
        test = gaussian(x,m,sig) - y
        if(np.abs(test) < cutoff):
            res=1
        else:
            res=0
        return res    
         
    # some parameters       
    m=0
    sig=0.5
    cutoff=0.1
    
    #print(points)   
    labels = np.zeros(len(points), dtype=np.int32)
        
    for n in range(0,len(points)):
        x=points[n,0]
        y=points[n,1]
        
        #print(x,y,gaussian(x,m,sig),target_marker(x,m,sig,y,cutoff))
        labels[n]=target_marker(x,m,sig,y,cutoff)
        
    return points, labels   
    
    
# In this version we prepare a dataset that has as many 0 as 1 label points    
def _gauss2(points):
      
    def gaussian(x,m,sig):
        norm = 0.9 # puts whole range into random grid ranges, should be this: 1/(sig * np.sqrt(2*np.pi)) but we want it between 0 and 1
        res = norm * np.exp(-0.5 * ((x-m)/sig)**2 )
        return res
    
    def target_marker(x,m,sig,y,cutoff):
        test = gaussian(x,m,sig) - y
        if(np.abs(test) < cutoff):
            res=1
        else:
            res=0
        return res    
         
    # some parameters       
    m=0
    sig=0.5
    cutoff=0.05 # defines width of Gaussian tube
    nratio=0.5 # controls the reatio between 0 and 1 labels, 1=random, 0=all on Gauss tube
    
    #print(points)   
    labels = np.zeros(len(points), dtype=np.int32)
    
    count=0    
    for n in range(0,len(points)):
        x=points[n,0]
        y=points[n,1]
        
        # check label first, if it's zero and there's less than len(points)/2 of them, take it
        tag=target_marker(x,m,sig,y,cutoff)
        if tag == 0 and count<int(len(points)*nratio):
            labels[n]=tag
            count+=1
        elif tag == 1: # keep the point if it's got a label in the Gaussian tube already
            labels[n]=tag 
        else: # create a random point at x on within the Gaussian tube
            y_new=gaussian(x,m,sig) + cutoff * ( 1 - 2 * np.random.rand(1, 1))   
            points[n,1]=y_new
            labels[n]=1
        
        
    return points, labels       
     



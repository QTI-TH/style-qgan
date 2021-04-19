import numpy as np
from itertools import product


def create_dataset(name, grid=None, samples=1000, seed=0):
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
        x = 1 - 2 * np.random.rand(samples, 1) # between -1 and 1
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
    elif name in ['gauss']:
        targets = [np.array([1, 0], dtype='complex'),
                   np.array([0, 1], dtype='complex')]

    return targets



def _circle(points):
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(np.linalg.norm(points, axis=1) > np.sqrt(2 / np.pi))
    labels[ids] = 1

    return points, labels
    
    
def _gauss(points):
      
    def gaussian(x,m,sig):
        norm = 1 # should be this: 1/(sig * np.sqrt(2*np.pi)) but we want it between 0 and 1
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
    sig=1
    cutoff=0.02
    
    #print(points)   
    labels = np.zeros(len(points), dtype=np.int32)
    
    for n in range(0,len(points)):
        x=points[n,0]
        y=points[n,1]
        
        #print(x,y,gaussian(x,m,sig),target_marker(x,m,sig,y,cutoff))
        labels[n]=target_marker(x,m,sig,y,cutoff)
        

    return points, labels    



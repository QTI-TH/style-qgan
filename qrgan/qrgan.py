# /usr/bin/env python
#import datasets as ds
import numpy as np
from qgenerator import single_qubit_generator

# ###############################################################################
#
# File qran.py
#
# Copyright (C) 2021 Anthony Francis
#
# This software is distributed under the terms of the GNU General Public
# License (GPL)
#
# Quantum re-uploading GAN
#
# 
# Code based on qibo qlassifier tutorial found at 
# https://qibo.readthedocs.io/en/stable/tutorials/reuploading_classifier/README.html
# Authors: Adrian Perez Salinas, Stefano Carrazza, Stavros Efthymiou
# Distributed under the apache 2 license.  
#
# ###############################################################################


# #########################
# The discriminator
# #########################

# For now the discriminator is handled analytically


# #########################
# The generator
# #########################

# need to tell the generator how many layers there were in the discriminator
dlayers=1 # hacked to be equal to qlassifier layers 

# can choose a different number of generator vs. discriminator layers
glayers=4

# train a new qgenerator? 1=yes, 0=no - requires parameters to be saved in out.qlassi.parameters
new_qgenerator=1

# Define generator
print("# ### Generator with {} layers".format(glayers))
qg = single_qubit_generator(glayers,dlayers)  

# Run generator training
if new_qgenerator==1:
    print('# Running new qgenerator parameters')
    
    # do not use the scipy minimize as before, the search space is too small
    #gresult, gparameters = qg.minimize(method='l-bfgs-b', options={'disp': True}) 
    
    # genetic algorithm seems to work better
    gresult, gparameters = qg.minimize(method='cma', options={'seed':113895, 'maxiter': 100})

    outf=open("./out.qgen.parameters", "w")
    
    for n in range(0,len(gparameters)):
        outf.write("%13e " %(gparameters[n]) )
    outf.flush()
    outf.close    
else:
    print('# Reading qgenerator parameters...')
    gparameters = np.loadtxt("./out.qgen.parameters"); 



# Output parameters and generate data      
print("# Parameters are:")
print(gparameters) 

qg.set_parameters(gparameters)
qg.generate()

value_loss = qg.cost_function()
print('# The value of the qgenerator cost function achieved is %.6f' % value_loss.numpy())

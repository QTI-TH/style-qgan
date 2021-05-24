# /usr/bin/env python
#import datasets as ds
import numpy as np
from qgenerator import single_qubit_generator
from qclassifier import single_qubit_classifier

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

# set discriminator layers
dlayers=4 

# train a new qgenerator? 1=yes, 0=no - requires parameters to be saved in out.qlassi.parameters
new_qdiscriminator=1

# Define discriminator
print("# ### Discriminator with {} layers".format(dlayers))
qd = single_qubit_classifier(dlayers)  

qd.predict()

# Run discriminator training
if new_qdiscriminator==1:
    print('# Running new discriminator parameters')
    
    # do not use the scipy minimize as before, the search space is too small
    #dresult, dparameters = qd.minimize(method='l-bfgs-b', options={'disp': True, 'maxiter': 3}) 
    
    # genetic algorithm seems to work better
    dresult, dparameters = qd.minimize(method='cma', options={'seed':113895, 'maxiter': 10})

    outf=open("./out.qdsc.parameters", "w")
    
    for n in range(0,len(dparameters)):
        outf.write("%13e " %(dparameters[n]) )
    outf.flush()
    outf.close    
else:
    print('# Reading qgenerator parameters...')
    gparameters = np.loadtxt("./out.qdsc.parameters"); 

# Output parameters and generate data      
print("# Parameters are:")
print(dparameters) 

qd.set_parameters(dparameters)
qd.predict()

value_loss = qd.cost_function()
print('# The value of the qdiscriminator cost function achieved is %.6f' % value_loss.numpy())



# #########################
# The generator
# #########################

# No discriminator, Kolmogorov-Smirnov instead

# generator layers, discriminator layer needs to be set irrespectively
glayers=1

# train a new qgenerator? 1=yes, 0=no - requires parameters to be saved in out.qlassi.parameters
new_qgenerator=0

# Define generator
print("# ### Generator with {} layers".format(glayers))
qg = single_qubit_generator(glayers,dlayers)  

# Run generator training
if new_qgenerator==1:
    print('# Running new qgenerator parameters')
    
    # do not use the scipy minimize as before, the search space is too small
    #gresult, gparameters = qg.minimize(method='l-bfgs-b', options={'disp': True, 'maxiter': 3}) 
    
    # genetic algorithm seems to work better
    gresult, gparameters = qg.minimize(method='cma', options={'seed':113895, 'maxiter': 3})

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

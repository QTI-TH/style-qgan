# /usr/bin/env python
import datasets_a as ds
import numpy as np
from qlassifier_a import single_qubit_classifier
from qgenerator_a import single_qubit_generator

# Code, based on qibo qlassifier tutorial



# #########################
# The discriminator
# #########################

# Some parameters and setup

layers=2
new_qlassifier=0 # train a new qlassifier? 1=yes, 0=no - requires parameters to be saved in out.qlassi.parameters

"""Perform classification for a given problem and number of layers.
"""
ql = single_qubit_classifier("gauss2",layers)  # Define classifier
ql.output()

# Train and run qlassifier as discriminator
if new_qlassifier==1:
    print('# Finding new qlassifier parameters')
    result, parameters = ql.minimize(method='l-bfgs-b', options={'disp': True})
    outf=open("./out.qlassi.parameters", "w")
    
    for n in range(0,len(parameters)):
        outf.write("%13e " %(parameters[n]) )
    outf.flush()
    outf.close    
    
else:
    print('# Reading qlassifier parameters...')
    parameters = np.loadtxt("./out.qlassi.parameters"); 
    
print("# Parameters are:")
print(parameters) 
 
ql.set_parameters(parameters)
ql.predict() 

value_loss = ql.cost_function_fidelity()
print('# The value of the qlassifier cost function achieved is %.6f' % value_loss.numpy())


# #########################
# The generator
# #########################

# Some parameters and setup

dlayers=2 # need to tell the generator how many layers there were in the discriminator
glayers=2 # can choose a different number of generator vs. discriminator layers
new_qgenerator=1 # train a new qgenerator? 1=yes, 0=no - requires parameters to be saved in out.qlassi.parameters

qg = single_qubit_generator("uniform",glayers,dlayers)  # Define generator

if new_qgenerator==1:
    print('# Running new qgenerator parameters')
    
    # do not use the minimize as before, the search space is too small
    #gresult, gparameters = qg.minimize(method='l-bfgs-b', options={'disp': True}) 
    
    # genetic algorithm
    gresult, gparameters = qg.minimize(method='cma', options={'disp': True})

    outf=open("./out.qgen.parameters", "w")
    
    for n in range(0,len(gparameters)):
        outf.write("%13e " %(gparameters[n]) )
    outf.flush()
    outf.close    
else:
    print('# Reading qgenerator parameters...')
    gparameters = np.loadtxt("./out.qgen.parameters"); 
    
print("# Parameters are:")
print(gparameters) 

qg.set_parameters(gparameters)
qg.generate()

value_loss = qg.cost_function_fidelity()
print('# The value of the qgenerator cost function achieved is %.6f' % value_loss.numpy())


# /usr/bin/env python
#import datasets as ds
import numpy as np
from qgenerator import single_qubit_generator
from qclassifier import single_qubit_classifier
import datasets as ds

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
# The GAN
# #########################

# set discriminator layers
dlayers=1

# set generator layers
glayers=1

# set size of mini batch
nmeas=50

# number of iterations in minimizer
maxiter=10

# set number of epochs
nepoch=100

# set up the generator and discriminator
qd = single_qubit_classifier(dlayers)
qg = single_qubit_generator(glayers,dlayers)


dseed=1
gseed=1
# loop over iterations
for n in range(0,nepoch):

    dseed+=1
    gseed+=1
    
    # create a sample of real data with labels=1
    xreal, yreal = ds.create_target_training('gauss',nmeas,dseed)

    # set the generator parameters from last iteration, except if it's the first iteration
    if n==0:
        gpar = qg.params
        dpar = qd.params
    else:
        qg.set_parameters(gpar)
        qd.set_parameters(dpar)
    
    # create a sample of fake data with labels=0    
    xinput = ds.create_dataset(nmeas,1,gseed)
    xfake = qg.generate(xinput,gpar)
    yfake = np.zeros(nmeas)

    # train the discriminator on these two samples, just one iteration of the minimizer

    # first the real
    qd.set_data(xreal,yreal)
    #dres_real, dpar = qd.minimize(method='cma', options={'verb_disp':0, 'seed':113895, 'maxiter': 2}) 
    dres_real, dpar = qd.minimize(method='l-bfgs-b', options={'disp': False, 'maxiter': maxiter}) 
    qd.set_parameters(dpar)
    #print("# Real train:", qd.params,dres_real)
    #qd.predict(xreal,dpar)
    #print("#        Real train got it right {}%".format( (1-np.sum(yreal-qd.predict(xreal,dpar)))*100 )  )
    rreal=0
    for i in range(0,len(yfake)):
        yguess=qd.predict(xreal,dpar)
        qtst=(yreal[0][i]-yguess[i])
        if qtst==0:
            rreal+=1

    # then the fake, the first guess parameters have been set by the previous training
    qd.set_data([xfake],[yfake])
    #dres_fake, dpar = qd.minimize(method='cma', options={'verb_disp':0, 'seed':113895, 'maxiter': 2}) 
    dres_fake, dpar = qd.minimize(method='l-bfgs-b', options={'disp': False, 'maxiter': maxiter}) 
    qd.set_parameters(dpar)
    #qd.predict([xfake],dpar)
    #print("#        Fake train got it right {}%".format( (1-np.sum(yfake-qd.predict([xfake],dpar)))*100 )  )
    rfake=0
    for i in range(0,len(yfake)):
        yguess=qd.predict([xfake],dpar)
        qtst=(yfake[i]-yguess[i])
        if qtst==0:
            rfake+=1
                        
    #print(rreal,rfake)
    print("# -------- Discriminator update, correct guess: Real {}/{}, Fake {}/{}".format(rreal,nmeas,rfake,nmeas))
        

    dloss= 0.5*(dres_real + dres_fake)
    #print("# Averaged discriminator loss:", dloss)


    # pass this info to the generator and minimize, just one iteration of the minimizer
    qg.set_dparameters(dpar)
    #print(qg.dparams, qg.params)
    qg.cost_function()
    #gres, gpar = qg.minimize(method='cma', options={'verb_disp':0, 'seed':113895, 'maxiter': 2}) 
    gres, gpar = qd.minimize(method='l-bfgs-b', options={'disp': False, 'maxiter': maxiter}) 
    
    # these are the new generator values, repeat the calculation
    qg.set_parameters(gpar)
    #print("# Real gen:", qg.params,gres)

    print("# Iteration {}: G_loss= {}, Davg_loss= {}".format(n,gres,dloss))
    

# #########################
# Finally, generate data
# #########################

print ("# Generate a few test samples")

nsamp=1000
nseed=nsamp

qg.set_parameters(gpar)
xinput = ds.create_dataset(nsamp,1,nseed)
xgen = qg.generate(xinput,gpar)

outf = open("./out.qgen.samples", "w")

for i in range(0,nsamp):
    outf.write("%.7e\n" % ( xgen[i] ))
        
outf.close
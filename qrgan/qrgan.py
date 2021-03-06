# /usr/bin/env python
#import datasets as ds
import numpy as np
from qgenerator import single_qubit_generator
from qclassifier import single_qubit_classifier
import datasets as ds
import tensorflow as tf

# ###############################################################################
#
# File qrgan.py
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
dlayers=10

# set generator layers
glayers=3

# set size of mini batch
nmeas=100

# number of iterations in minimizer
dmaxiter=10
gmaxiter=10

# set number of epochs and ksteps
nepoch=10
kwarm=2
krun=1

# set up the generator and discriminator
qd = single_qubit_classifier(dlayers,"fidelity")
qg = single_qubit_generator(glayers,dlayers)

# initiate seeds
dseed=1
gseed=2
fseed=3

# open losses file freshly
outf1 = open("./out.qgen.losses", "w")
outf1.close

# output target distribution
xtarget, ytarget = ds.create_target_training('gauss',1000,dseed)
outf = open("./out.qgen.target", "w")
for i in range(0,1000):
    outf.write("%.7e\n" % ( xtarget[0][i] ))        
outf.close

# loop over iterations
for n in range(0,nepoch+1):
    
    # set the generator to initial
    if n==0:
        print("# Setting initial parameters")
        gpar = qg.params
        dpar = qd.params
    
    if n==0:
        kstep=kwarm
        print("# Warm up the discriminator: {} iterations".format(kwarm))
    else:
        kstep=krun
    
    for k in range(0,kstep):
    
        dseed+=1
        gseed+=1
        fseed+=1
    
        # create a mini-batch of real data with labels=1
        xreal, yreal = ds.create_target_training('gauss',nmeas,dseed)

        # set the generator parameters from last iteration or initial
        qg.set_parameters(gpar)
        qd.set_parameters(dpar)
    
        # create a sample of fake data with labels=0    
        xinput = ds.create_dataset('uniform_prior',nmeas,1,gseed)
        xfake = qg.generate(xinput,gpar)
        yfake = np.zeros(nmeas)

        # train the discriminator on these two samples

        # set the real and fake data, then update in joined cost function
        qd.set_data(xreal,yreal)
        qd.set_fake([xfake],[yfake])
        #dloss, dpar = qd.minimize(method='cma', options={'verb_disp':0, 'seed':113895, 'maxiter': dmaxiter}) 
        #dloss, dpar = qd.minimize(method='cma', options={'verb_disp':0, 'maxiter': dmaxiter}) 
        dloss, dpar = qd.minimize(method='l-bfgs-b', options={'disp': False, 'maxiter': dmaxiter}) 
        qd.set_parameters(dpar)

        # figure out how many times it managed to make the label be the passed one
        rfake=0
        yguess=qd.predict([xfake],dpar)
        for i in range(0,nmeas): 
            qtst=(yfake[i]-yguess[i])
            if qtst==0:
                rfake+=1           
        rreal=0
        yguess=qd.predict(xreal,dpar)
        for i in range(0,nmeas): 
            qtst=(yreal[0][i]-yguess[i])
            if qtst==0:
                rreal+=1        
        print("#              Discriminator update, correct guess: Real {} / {}, Fake {} / {} ".format(rreal,nmeas,rfake,nmeas))

        
    # pass this info to the generator and minimize
    qg.set_dparameters(dpar)
    qg.set_seed(fseed)
    qg.cost_function()
    #gloss, gpar = qg.minimize(method='cma', options={'verb_disp':0, 'seed':113895, 'maxiter': gmaxiter}) 
    gloss, gpar = qg.minimize(method='cma', options={'verb_disp':0, 'maxiter': gmaxiter}) 
    #gloss, gpar = qg.minimize(method='l-bfgs-b', options={'disp': True, 'maxiter': gmaxiter}) 
    
    # these are the new generator values, repeat the calculation
    qg.set_parameters(gpar)
    
    # figure out how many times the generator passed
    xinput = ds.create_dataset('uniform_prior',nmeas,1,fseed)
    xtest = qg.generate(xinput,gpar)
    #ytest = np.sum(qd.predict([xtest])) # can also use the discriminator as cross-check
    ytest = np.sum(qg.dpredict([xtest]))
    print("#              Generator update, times passed: {} / {}".format(int(ytest),len(xtest)))
    
    
    print("# Iteration {}: G_loss= {}, Davg_loss= {}".format(n,gloss,dloss))

    dres_real=rreal/nmeas
    dres_fake=rfake/nmeas
    dres=0.5*(dres_real+dres_fake)
    gres=ytest/len(xtest)
    
    outf1 = open("./out.qgen.losses", "a")
    outf1.write("%d  %.7e %.7e  %.2e %.2e\n" % ( n,gloss,dloss,gres,dres ))
    outf1.close    

    # #########################
    # generate data on the fly
    # #########################

    if n%2==0:
        #print ("# ============ Generate a few test samples")
        print ("#              Generate a few test samples")

        nsamp=1000
        nseed=nsamp

        qg.set_parameters(gpar)
        xinput = ds.create_dataset('uniform_prior',nsamp,1,nseed)
        xgen = qg.generate(xinput,gpar)

        outf2 = open("./out.qgen.samples.n{}".format(n), "w")

        for i in range(0,nsamp):
            outf2.write("%.7e\n" % ( xgen[i] ))
        
        outf2.close




# #########################
# Final data generation
# #########################

print ("# Generate a few test samples")

nsamp=1000
nseed=nsamp

qg.set_parameters(gpar)
xinput = ds.create_dataset('uniform_prior',nsamp,1,nseed)
xgen = qg.generate(xinput,gpar)

outf3 = open("./out.qgen.samples", "w")

for i in range(0,nsamp):
    outf3.write("%.7e\n" % ( xgen[i] ))
        
outf3.close
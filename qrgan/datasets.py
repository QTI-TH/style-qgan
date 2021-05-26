# ###############################################################################
#
# File datasets.py
#
# Copyright (C) 2021 Anthony Francis
#
# This software is distributed under the terms of the GNU General Public
# License (GPL)
#
# Routines to generate training data sets. Currently implemented:
# 
# Code based on qibo qlassifier tutorial found at 
# https://qibo.readthedocs.io/en/stable/tutorials/reuploading_classifier/README.html
# Authors: Adrian Perez Salinas, Stefano Carrazza, Stavros Efthymiou
# Distributed under the apache 2 license.    
#
# ###############################################################################
import numpy as np
from itertools import product

# create set of samples, nsamp=number of samples in set, nmeas=number of points in sample
def create_dataset(name, nmeas=10, nsamp=100, seed=0):
    
    data=np.zeros((nsamp,nmeas)) 
    
    if name in ['uniform_prior']:
        np.random.seed(seed)
        # draw from a uniform distribution
        #xwindow=1
        #data=xwindow * ( 1 - 2 * np.random.rand(nsamp, nmeas)) # between -xwindow and xwindow
        data=np.random.rand(nsamp, nmeas) # between 0 and 1
    
    if name in ['gauss_prior']:
        mf=float(np.random.rand(1, 1)) # between 0 and 1
        sigf=float(np.random.rand(1, 1)+0.01) # between 0.01 and swindow+0.01
        data=[np.random.default_rng(seed).normal(mf, sigf, nmeas)] # needs brackets because the uniform returns an array

    if name in ['gauss_prior_fix']:
        mf=0.5 
        sigf=0.5 
        data=[np.random.default_rng(seed).normal(mf, sigf, nmeas)] # needs brackets because the uniform returns an array
        
    return data

   

# Create a mini batch with labels    
def create_target_training(name, nmeas=10, nsamp=1, seed=0):
    
    data=np.zeros((nsamp,nmeas)) 
    labl=np.zeros((nsamp,nmeas)) 
        
    # draw from a Gaussian distribution
    if name in ['gauss']:
        # target distribution parameters
        m=0.5
        sig=0.125
        
        # draw nsamples of nmeas length    
        for n in range(0,nsamp): 
            seed+=1
            data[n,:] = np.random.default_rng(seed).normal(m, sig, nmeas)
            labl[n,:]=1
 
    # draw from a Log-normal distribution
    if name in ['lognormal']:
        # target distribution parameters
        m=-0.5
        sig=0.5
        
        # draw nsamples of nmeas length    
        for n in range(0,nsamp): 
            seed+=1
            data[n,:] = np.random.default_rng(seed).lognormal(m, sig, nmeas) # between 0 and 1, otherwise do -1 to expression
            labl[n,:]=1
         
    
    return data, labl
    





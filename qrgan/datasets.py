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
def create_dataset(nmeas=10, nsamp=100, seed=0):
    
    data=np.zeros((nsamp,nmeas)) 
    # draw from a uniform distribution
    xwindow=1
    data=xwindow * ( 1 - 2 * np.random.rand(nsamp, nmeas)) # between -xwindow and xwindow
    
    return data


# create one sample of the target distribution
def create_target(name, nmeas=1000, seed=0):
    
    
    # draw from a Gaussian distribution
    if name in ['gauss']:
        m=0
        sig=0.25
        target = np.random.default_rng(seed).normal(m, sig, nmeas)
    
        print("# Testing mean of target (Gaussian):")
        print(abs(m - np.mean(target)))
    
    # draw from a Log-normal distribution    
    if name in ['lognormal']:    
        m=-0.5
        sig=0.5
        target = np.random.default_rng(seed).lognormal(m, sig, nmeas)-1
        
        print("# Testing mean of target (Lognormal):")
        print(abs(m - np.mean(target)))
        
    
    return target
    

# Create a mini batch with labels    
def create_target_training(name, nmeas=10, nsamp=1, seed=0):
    
    data=np.zeros((nsamp,nmeas)) 
    labl=np.zeros((nsamp,nmeas)) 
        
    # draw from a Gaussian distribution
    if name in ['gauss']:
        # target distribution parameters
        m=0
        sig=0.25
        
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
            data[n,:] = np.random.default_rng(seed).lognormal(m, sig, nmeas)-1
            labl[n,:]=1
         
    
    return data, labl
    


# Discriminator training
# - create set of samples, nsamp=number of samples in set, nmeas=number of points in sample
# - in this set of samples there will be N samples of the target (true) and M false samples
#
# false samples are drawn from:
#       target = gauss: uniform and lognormal - only option right now
#       target = lognormal: uniform and gauss
#
def create_training(name, nmeas=10, nsamp=100, seed=0):
    
    data=np.zeros((nsamp,nmeas)) 
    labl=np.zeros(nsamp) 
    
    # half the samples are from the true distribution
    N=int(nsamp/2)
    M=nsamp
    
    # target distribution parameters
    m=0
    sig=0.25
    
    # draw from a Gaussian distribution
    if name in ['gauss']:
       
        # true samples
        for n in range(0,N): 
            seed+=1
            
            # draw from a Gaussian distribution
            data[n,:] = np.random.default_rng(seed).normal(m, sig, nmeas)
            labl[n]=1
        
        # false samples
        for n in range(N,M): 
            seed+=1
            
            if n%3==0:
                # draw from a uniform distribution
                xwindow=2
                np.random.seed(seed)
                data[n,:] = xwindow * ( 1 - 2 * np.random.rand(1, nmeas)) # between -xwindow and xwindow
                labl[n]=0
            elif n%3==1:
                # draw from a (random) log normal distribution
                np.random.seed(seed)
                mwindow=2
                swindow=2
                mf=float(mwindow * ( 1 - np.random.rand(1, 1))) # between 0 and mwindow
                sigf=float(swindow * ( 1.1 - np.random.rand(1, 1))) # between 0.1 and swindow+0.1
                data[n,:] = np.random.default_rng(seed).lognormal(mf, sigf, nmeas)-1
                labl[n]=0
            elif n%3==2:
                # draw from a (random) normal distribution
                np.random.seed(seed)
                mwindow=2
                swindow=2
                mf=float(mwindow * ( 1.1 - np.random.rand(1, 1))) # between 0.1 and mwindow+0.1
                sigf=float(swindow * ( 1.1 - np.random.rand(1, 1))) # between 0.1 and swindow+0.1
                data[n,:] = np.random.default_rng(seed).normal(mf, sigf, nmeas)
                labl[n]=0
                
                
        
        #print(data, labl)
        
    
    return data, labl



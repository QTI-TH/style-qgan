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
def create_dataset(nmeas=10, nsamp=200, seed=0):
    
    data=np.zeros((nsamp,nmeas)) 
    # draw from a uniform distribution
    xwindow=1
    data=xwindow * ( 1 - 2 * np.random.rand(nsamp, nmeas)) # between -xwindow and xwindow
    
    return data


# create one sample of the target distribution
def create_target(nmeas=2000, seed=0):
    
    
    # draw from a Gaussian distribution
    m=0
    sig=0.25
    target = np.random.default_rng(seed).normal(m, sig, nmeas)
    
    print("# Testing mean of target:")
    print(abs(m - np.mean(target)))
    
    return target




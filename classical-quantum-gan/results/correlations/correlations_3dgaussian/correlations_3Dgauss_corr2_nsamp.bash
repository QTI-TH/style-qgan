#!/bin/bash

SAMPLES='1000 5000 10000 20000'
for samples in $SAMPLES
do

case $samples in
1000) smp=1k;;
5000) smp=5k;;
10000) smp=10k;;
20000) smp=20k;;
esac

LAY='2 5 7 10'
#LAY='5'
for lay in $LAY
do
	
	
rm measure.corr_${lay}_${smp}
rm measure.correig_${lay}_${smp}

for (( lat=1; lat<=6; lat++))
do	

python3.8 - << MARKER
import numpy as np
import os
import time
from math import sqrt
from math import pi
from random import randint
from random import seed
from scipy import linalg as LA
from scipy.optimize import curve_fit
from scipy import optimize
import pandas as pd
from scipy.stats import entropy
from scipy.stats import kstest
import scipy as sc
from scipy.stats import norm

# ### Define iterator

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

# ### output


nqubits = 3
latent_dim = $lat #5
layers = $lay #2
samples = $samples

print("# --------------------------------------------------------------------------------------- ")
print("Checking Nqubits={}, latent_dim={}, layers={}, samples={}".format(nqubits,latent_dim,layers,samples))


# ### exact setting

# there is a factor 4*4=16 between the written covm and the determined covmatt, due to the 1/4. normalisation when drawing x
covm = [[0.5, 0.1, 0.25], [0.1, 0.5, 0.1], [0.25, 0.1, 0.5]] 
mean = [0, 0, 0]
x = np.random.multivariate_normal(mean, covm, ${samples}).T/4
#print(x)
#print(covm)

w0, v0 = LA.eig(np.cov(x))
#print(w0.real)

# ### data

dat1 = np.loadtxt(f"qgan${smp}/3dgaussian_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(0,)); 
dat2 = np.loadtxt(f"qgan${smp}/3dgaussian_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(1,)); 
dat3 = np.loadtxt(f"qgan${smp}/3dgaussian_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(0,1,2)); 
dat4 = np.loadtxt(f"qgan${smp}/3dgaussian_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(3,4,5)); 

# there is a factor 4*4=16 between the written covm and the determined covmatt, due to the 1/4. normalisation when drawing x
covmatt=(np.cov(dat3.T))
covmatt2=np.cov(x)
#print(covmatt)

# exact resampled
#print("  Difference between exact covmat and after sampling, sum=",np.sum(covmatt2-covmatt),"and matrix:")
#print(covmatt2-covmatt)

print("  Difference between exact covmat and after sampling, sum=",np.sum(covm-covmatt*16),"and matrix:")
print(covm-covmatt*16)

dmeas3=np.sum(covmatt*16/covm)/9.
print("  Relative difference exact and target covmat, dmeas=",dmeas3,"and matrix:")
print(covmatt*16/covm)


covmat=(np.cov(dat4.T))
#print(covmat)

print("  Difference of target and learned covmat, sum=",np.sum(covmatt-covmat),"and matrix:")
print(covmatt-covmat)
#print(covmatt)
#print(covmat)
dmeas=np.sum(covmat/covmatt)/9.
print("  Relative difference target and learned covmat, dmeas=",dmeas,"and matrix:")
print(covmat/covmatt)


dmeas2=np.sum(covmat*16/covm)/9.
print("  Relative difference exact and learned covmat, dmeas=",dmeas2,"and matrix:")
print(covmat*16/covm)

# ####

w0,v0 = LA.eig(covm)
wt,vt = LA.eig(covmatt)
w,v = LA.eig(covmat)
print("  Eigenvalues of target and learned covmat:", np.sort(wt.real),np.sort(w.real), np.sort(w0.real/16))

print(" Averaged ratio of Eigenvalues (target, fake):", np.sum(np.sort(wt.real)/np.sort(w.real))/3.)
print(" Averaged ratio of Eigenvalues (target, exact):", np.sum(np.sort(wt.real)/np.sort(w0.real/16))/3.)
print(" Averaged ratio of Eigenvalues (exact, fake):", np.sum(np.sort(w0.real/16)/np.sort(w.real))/3.)
outf3 = open("measure.correig_${lay}_${smp}", "a")
outf3.write("%d %d %d  %.13e %.13e %.13e   %.13e %.13e %.13e\n" % (nqubits,latent_dim,layers,np.sum(np.sort(wt.real)/np.sort(w.real))/3., np.sum(np.sort(wt.real)/np.sort(w0.real/16))/3., np.sum(np.sort(w0.real/16)/np.sort(w.real))/3.,dmeas,dmeas3,dmeas2))
outf3.close



# ####

# Generate samples from three independent normally distributed random
# variables (with mean 0 and std. dev. 1).
x = norm.rvs(size=(3, ${samples}), loc=0, scale=0.5)
		
# Convert the data to correlated random variables. 
y0 = np.dot(np.linalg.cholesky(covm), x)
y1 = np.dot(np.linalg.cholesky(covmatt), x)
y2 = np.dot(np.linalg.cholesky(covmat), x)

#print(y0)

#y1=dat3.T

offset=1
kl_divergence1 = np.around(0.5* (entropy(y1[0]+offset, y2[0]+offset) + entropy(y1[0]+offset, y2[0]+offset)) , 13) 
kl_divergence2 = np.around(0.5* (entropy(y1[1]+offset, y2[1]+offset) + entropy(y1[1]+offset, y2[1]+offset)) , 13) 
kl_divergence3 = np.around(0.5* (entropy(y1[2]+offset, y2[2]+offset) + entropy(y1[2]+offset, y2[2]+offset)) , 13) 
print("  KL divergences in (x,x), (y,y) and (z,z):",kl_divergence1,kl_divergence2,kl_divergence3)
kl_sum=kl_divergence1+kl_divergence2+kl_divergence3

outf3 = open("measure.corr_${lay}_${smp}", "a")
outf3.write("%d %d %d %.13e %.13e %.13e  %.13e\n" % (nqubits,latent_dim,layers,kl_divergence1,kl_divergence2,kl_divergence3,kl_sum))
outf3.close


MARKER


done
done
done

exit 0



#!/bin/bash



LAY='2'
LAT='5'

for lay in $LAY
do
	
rm measure.aws.corr_${lay}
rm measure.aws.correig_${lay}

for lat in $LAT
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


nbin=40
nqubits = 3
latent_dim = $lat #5
layers = $lay #2

print("# --------------------------------------------------------------------------------------- ")
print("Checking Nqubits={}, latent_dim={}, layers={}".format(nqubits,latent_dim,layers))


# ### data

dat3 = np.loadtxt(f"qgan_aws/lhc_ttbar_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(0,1,2)); 
dat3a = np.loadtxt(f"qgan/lhc_ttbar_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(0,1,2)); 
dat3b= np.loadtxt(f"qgan_aws/lhc_ttbar_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(3,4,5)); 
dat4 = np.loadtxt(f"qgan_aws/out_aws_generator_samples.txt",usecols=(0,1,2)); 

covmatt=(np.cov(dat3.T))
covmatt10=(np.cov(dat3a.T))
covmatf=(np.cov(dat3b.T))
covmat=(np.cov(dat4.T))

print("  Estimated target covmat 10k:")
print(covmatt10)

print("  Estimated target covmat:")
print(covmatt)

dmeas=np.sum(covmatt/covmatt10)/9.
print("  Relative difference 10k and 1k target covmats, dmeas=",dmeas,"and matrix:")
print(covmatt/covmatt10)

print("  Obtained learned covmat (simulated):")
print(covmatf)

print("  Obtained learned covmat (aws):")
print(covmat)

#print("  Difference of target and learned covmat, sum=",np.sum(covmatt-covmat),"and matrix:")
#print(covmatt-covmat)

dmeas=np.sum(covmat/covmatt)/9.
print("  Relative difference target and learned aws covmat, dmeas=",dmeas,"and matrix:")
print(covmat/covmatt)

dmeas=np.sum(covmat/covmatf)/9.
print("  Relative difference learned simulated and aws covmat , dmeas=",dmeas,"and matrix:")
print(covmat/covmatf)
# ####

wt10,vt10 = LA.eig(covmatt10)
wt,vt = LA.eig(covmatt)
wf,vf = LA.eig(covmatf)
w,v = LA.eig(covmat)
print("  Eigenvalues of target 10k, 1k, simulated and aws covmat:", np.sort(wt10.real),np.sort(wt.real),np.sort(wf.real),np.sort(w.real))

print(" Averaged ratio of Eigenvalues (target 10k/1k):", np.sum(np.sort(wt10.real)/np.sort(wt.real))/3.)
print(" Averaged ratio of Eigenvalues (target 10k/simulated):", np.sum(np.sort(wt10.real)/np.sort(wf.real))/3.)
print(" Averaged ratio of Eigenvalues (target 10k/aws):", np.sum(np.sort(wt10.real)/np.sort(w.real))/3.)
print(" Averaged ratio of Eigenvalues (target 1k/simulated):", np.sum(np.sort(wt.real)/np.sort(wf.real))/3.)
print(" Averaged ratio of Eigenvalues (target 1k/aws):", np.sum(np.sort(wt.real)/np.sort(w.real))/3.)
print(" Averaged ratio of Eigenvalues (simulated/aws):", np.sum(np.sort(wf.real)/np.sort(w.real))/3.)


outf3 = open("measure.aws.correig_${lay}", "a")
outf3.write("%d %d %d  %.13e   %.13e\n" % (nqubits,latent_dim,layers,np.sum(np.sort(wt.real)/np.sort(w.real))/3.,dmeas))
outf3.close



# ####

# Generate samples from three independent normally distributed random
# variables (with mean 0 and std. dev. 1).
x = norm.rvs(size=(3, 10000), loc=0, scale=0.5)
		
# Convert the data to correlated random variables. 
y1 = np.dot(np.linalg.cholesky(covmatt), x)
y2 = np.dot(np.linalg.cholesky(covmat), x)

#print(y0)

#y1=dat3.T

offset=1000
kl_divergence1 = np.around(0.5* (entropy(y1[0]+offset, y2[0]+offset) + entropy(y1[0]+offset, y2[0]+offset)) , 13) 
kl_divergence2 = np.around(0.5* (entropy(y1[1]+offset, y2[1]+offset) + entropy(y1[1]+offset, y2[1]+offset)) , 13) 
kl_divergence3 = np.around(0.5* (entropy(y1[2]+offset, y2[2]+offset) + entropy(y1[2]+offset, y2[2]+offset)) , 13) 
print("  KL divergences in (x,x), (y,y) and (z,z):",kl_divergence1,kl_divergence2,kl_divergence3)
kl_sum=kl_divergence1+kl_divergence2+kl_divergence3

outf3 = open("measure.aws.corr_${lay}", "a")
outf3.write("%d %d %d %.13e %.13e %.13e  %.13e\n" % (nqubits,latent_dim,layers,kl_divergence1,kl_divergence2,kl_divergence3,kl_sum))
outf3.close


MARKER


done
done


exit 0

rm measure.corr2


SET='1 2 3 4 5 6'
SET='7 8 9 10 11 12'
SET='7 8 9 10 11 12 13 14 15'
for set in $SET
do
	
	case $set in
		1) lat=1; lay=2;;
		2) lat=2; lay=2;;
		3) lat=3; lay=2;;
		4) lat=4; lay=2;;
		5) lat=5; lay=2;;
		6) lat=6; lay=2;;
		7) lat=1; lay=4;;
		8) lat=2; lay=4;;
		9) lat=3; lay=4;;
		10) lat=4; lay=4;;
		11) lat=5; lay=4;;
		12) lat=6; lay=4;;
		13) lat=7; lay=4;;
		14) lat=8; lay=4;;
		15) lat=9; lay=4;;
	esac	



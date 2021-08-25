#!/bin/bash

	


LAY='2 4'
LAT='5'

for lay in $LAY
do
	
rm measure.veig2_3D_${lay}

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
nbin2=nbin #50

print("Checking Nqubits={}, latent_dim={}, layers={}".format(nqubits,latent_dim,layers))

# ### data

dat1 = np.loadtxt(f"./results/target_{nqubits}_{latent_dim}_{layers}.3dhist",usecols=(3,)); 
dat1.shape = (nbin,nbin,nbin)

mmat = np.zeros((nbin,nbin,nbin))
mmat=dat1				
mt1 = np.zeros((nbin))	
mt2 = np.zeros((nbin))	
mt3 = np.zeros((nbin))	
for b1 in range(0,nbin):
	for b2 in range(0,nbin):
		for b3 in range(0,nbin):
			mt1[b1]+=mmat[b1,b2,b3]
			mt2[b1]+=mmat[b2,b1,b3]
			mt3[b1]+=mmat[b2,b3,b1]

		

# ### fake data

dat1 = np.loadtxt(f"./results/qc_{nqubits}_{latent_dim}_{layers}.3dhist",usecols=(3,)); 
dat1.shape = (nbin,nbin,nbin)

mmat = np.zeros((nbin,nbin,nbin))
mmat=dat1				
m1 = np.zeros((nbin))	
m2 = np.zeros((nbin))	
m3 = np.zeros((nbin))	
for b1 in range(0,nbin):
	for b2 in range(0,nbin):
		for b3 in range(0,nbin):
			m1[b1]+=mmat[b1,b2,b3]
			m2[b1]+=mmat[b2,b1,b3]
			m3[b1]+=mmat[b2,b3,b1]



# ### KL divergences


offset=50
ks_test = np.around(kstest(mt1,m1), 3)
kl_divergence = np.around(0.5* (entropy(mt1+offset, m1+offset) + entropy(m1+offset, mt1+offset)) , 3) 
#print(ks_test,kl_divergence)

kl_divergence1 = np.around(0.5* (entropy(mt1+offset, m1+offset) + entropy(m1+offset, mt1+offset)) , 3) 
kl_divergence2 = np.around(0.5* (entropy(mt2+offset, m2+offset) + entropy(m2+offset, mt2+offset)) , 3) 
kl_divergence3 = np.around(0.5* (entropy(mt3+offset, m3+offset) + entropy(m3+offset, mt3+offset)) , 3) 
print(kl_divergence1,kl_divergence2,kl_divergence3)

print(kl_divergence1+kl_divergence2+kl_divergence3)

outf3 = open("measure.veig2_3D_${lay}", "a")
outf3.write("%d %d %d %.3f %.3f %.3f\n" % (nqubits,latent_dim,layers,kl_divergence1,kl_divergence2,kl_divergence3))
outf3.close


MARKER


done
done


exit 0



SET='1 2 3 4 5 6 7 8 9 10'
#SET='1'
for set in $SET
do
	
	case $set in
		1) lat=4; lay=1;;
		2) lat=8; lay=2;;
		3) lat=5; lay=1;;
		4) lat=5; lay=2;;
		5) lat=5; lay=5;;
		6) lat=1; lay=10;;
		7) lat=2; lay=10;;
		8) lat=3; lay=10;;
		9) lat=4; lay=10;;
		10) lat=5; lay=10;;
		11) lat=6; lay=10;;
	esac	
done



# Generate samples from three independent normally distributed random
# variables (with mean 0 and std. dev. 1).
x = norm.rvs(size=(20, 1))
		
# Convert the data to correlated random variables. 
y = np.dot(mmat, x)

print(y)




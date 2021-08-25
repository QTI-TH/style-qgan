#!/bin/bash






LAY='2 5 7 10'
for lay in $LAY
do

rm measure.3dhist_${lay}

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

print("Checking Nqubits={}, latent_dim={}, layers={}".format(nqubits,latent_dim,layers))


# ### data

dat1 = np.loadtxt(f"qgan10k/3dgaussian_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(0,)); 
dat2 = np.loadtxt(f"qgan10k/3dgaussian_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(1,)); 
dat3 = np.loadtxt(f"qgan10k/3dgaussian_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(2,)); 

R1 = np.histogramdd(np.array([dat1,dat2,dat3]).T,bins=nbin,range=[[-1,1],[-1,1],[-1,1]])
xarr=R1[1][0]
yarr=R1[1][1]
zarr=R1[1][2]
aval=R1[0]

outf1 = open(f"./results/target_{nqubits}_{latent_dim}_{layers}.3dhist", "w")
for b1 in range(0,nbin):
	for b2 in range(0,nbin):
		for b3 in range(0,nbin):
			outf1.write("%.3e %.3e %.3e  %d \n" % (xarr[b1],yarr[b2],zarr[b3],aval[b1,b2,b3]))
outf1.close

outf1 = open(f"./results/target_{nqubits}_{latent_dim}_{layers}.1dhist1", "w")
fhist , fedge =np.histogram(dat1,bins=nbin,range=(-1,1))
for b in range(0,nbin):
	outf1.write("%.14e %d\n" % (fedge[b],fhist[b]))
outf1.close

outf1 = open(f"./results/target_{nqubits}_{latent_dim}_{layers}.1dhist2", "w")
fhist , fedge =np.histogram(dat2,bins=nbin,range=(-1,1))
for b in range(0,nbin):
	outf1.write("%.14e %d\n" % (fedge[b],fhist[b]))
outf1.close

outf1 = open(f"./results/target_{nqubits}_{latent_dim}_{layers}.1dhist3", "w")
fhist , fedge =np.histogram(dat3,bins=nbin,range=(-1,1))
for b in range(0,nbin):
	outf1.write("%.14e %d\n" % (fedge[b],fhist[b]))
outf1.close


Rt1 = np.histogram2d(dat1,dat2,bins=nbin,range=[[-1,1],[-1,1]])
Rt2 = np.histogram2d(dat1,dat3,bins=nbin,range=[[-1,1],[-1,1]])
Rt3 = np.histogram2d(dat2,dat3,bins=nbin,range=[[-1,1],[-1,1]])
Rtv1=Rt1[0]
Rtv2=Rt2[0]
Rtv3=Rt3[0]


# #### fake data

dat1 = np.loadtxt(f"qgan10k/3dgaussian_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(3,)); 
dat2 = np.loadtxt(f"qgan10k/3dgaussian_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(4,)); 
dat3 = np.loadtxt(f"qgan10k/3dgaussian_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(5,)); 

R1 = np.histogramdd(np.array([dat1,dat2,dat3]).T,bins=nbin,range=[[-1,1],[-1,1],[-1,1]])
xarr=R1[1][0]
yarr=R1[1][1]
zarr=R1[1][2]
aval2=R1[0]

outf1 = open(f"./results/qc_{nqubits}_{latent_dim}_{layers}.3dhist", "w")
for b1 in range(0,nbin):
	for b2 in range(0,nbin):
		for b3 in range(0,nbin):
			outf1.write("%.3e %.3e %.3e  %d  %d \n" % (xarr[b1],yarr[b2],zarr[b3],aval2[b1,b2,b3],aval[b1,b2,b3]))
outf1.close


Rf1 = np.histogram2d(dat1,dat2,bins=nbin,range=[[-1,1],[-1,1]])
Rf2 = np.histogram2d(dat1,dat3,bins=nbin,range=[[-1,1],[-1,1]])
Rf3 = np.histogram2d(dat2,dat3,bins=nbin,range=[[-1,1],[-1,1]])
Rfv1=Rf1[0]
Rfv2=Rf2[0]
Rfv3=Rf3[0]

outf1 = open(f"./results/qc_{nqubits}_{latent_dim}_{layers}.2dsubhist", "w")
for b1 in range(0,nbin):
	for b2 in range(0,nbin):
			outf1.write("%.3e %.3e  %d %d %d   %d %d %d\n" % (xarr[b1],yarr[b2],Rtv1[b1,b2],Rtv2[b1,b2],Rtv3[b1,b2],Rfv1[b1,b2],Rfv2[b1,b2],Rfv3[b1,b2]))
outf1.close			

# ### define real sample generator for baseline and error

covm = [[0.5, 0.1, 0.25], [0.1, 0.5, 0.1], [0.25, 0.1, 0.5]] 
mean = [0, 0, 0]

nerr=50
merr=np.zeros(nerr)
for ierr in range(0,nerr):
	x = np.random.multivariate_normal(mean, covm, 10000).T/4
	rerr, ferr = np.histogramdd(x.T,bins=nbin,range=[[-1,1],[-1,1],[-1,1]])
	merr[ierr]= (np.sum(np.abs(rerr-aval)))/nbin**2

mmerr=np.around(np.mean(merr),5)
mmerr2=np.around(np.std(merr),5)

meas= (np.sum(np.abs(aval-aval2)))/nbin**2
mmeas=np.around(meas/mmerr,5)

#mdat=-merr+meas
#print(mdat, np.around(np.mean(mdat),3),  np.around(np.std(mdat),3))
#print(np.around(np.mean(mdat),3),  np.around(np.std(mdat),3))

#print(aval-aval2,meas)
print(meas,mmerr,mmerr2,mmeas)

outf3 = open("measure.3dhist_${lay}", "a")
outf3.write("%d %d %d  %.3f %.3f\n" % (nqubits,latent_dim,layers,mmeas,mmerr2))
outf3.close

#R2 = np.cov(dat1)
#print(R2)

#R3 = np.corrcoef(R2)
#print(R3)

MARKER

done
done


exit 0






# gnuplot
set view map
set pm3d
set dgrid3d
set pm3d interpolate 1,1
splot "qc.2dhist" using 1:2:3 p3md
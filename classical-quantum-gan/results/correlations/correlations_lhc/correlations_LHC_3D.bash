#!/bin/bash




LAY='2 4'
LAT='5'

for lay in $LAY
do
	
rm measure.3dhist_${lay}

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

dat1 = np.loadtxt(f"qgan/lhc_ttbar_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(0,)); 
dat2 = np.loadtxt(f"qgan/lhc_ttbar_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(1,)); 
dat3 = np.loadtxt(f"qgan/lhc_ttbar_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(2,)); 

R1 = np.histogramdd(np.array([dat1,dat2,dat3]).T,bins=nbin,range=[[0,9e5],[-6e5,0],[-4,4]])
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


Rt1 = np.histogram2d(dat1,dat2,bins=nbin,range=[[0,9e5],[-6e5,0]])
Rt2 = np.histogram2d(dat1,dat3,bins=nbin,range=[[0,9e5],[-4,4]])
Rt3 = np.histogram2d(dat2,dat3,bins=nbin,range=[[-6e5,0],[-4,4]])
Rtv1=Rt1[0]
Rtv2=Rt2[0]
Rtv3=Rt3[0]

xarr1=Rt1[1]
yarr1=Rt1[2]

xarr2=Rt2[1]
yarr2=Rt2[2]

xarr3=Rt3[1]
yarr3=Rt3[2]


# #### fake data

dat1 = np.loadtxt(f"qgan/lhc_ttbar_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(3,)); 
dat2 = np.loadtxt(f"qgan/lhc_ttbar_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(4,)); 
dat3 = np.loadtxt(f"qgan/lhc_ttbar_circuit_{nqubits}_{latent_dim}_{layers}.smp",usecols=(5,)); 

R1 = np.histogramdd(np.array([dat1,dat2,dat3]).T,bins=nbin,range=[[0,9e5],[-6e5,0],[-4,4]])
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


Rf1 = np.histogram2d(dat1,dat2,bins=nbin,range=[[0,9e5],[-6e5,0]])
Rf2 = np.histogram2d(dat1,dat3,bins=nbin,range=[[0,9e5],[-4,4]])
Rf3 = np.histogram2d(dat2,dat3,bins=nbin,range=[[-6e5,0],[-4,4]])
Rfv1=Rf1[0]
Rfv2=Rf2[0]
Rfv3=Rf3[0]

outf1a = open(f"./results/qc_{nqubits}_{latent_dim}_{layers}.2dsubhist1", "w")
outf1b = open(f"./results/qc_{nqubits}_{latent_dim}_{layers}.2dsubhist2", "w")
outf1c = open(f"./results/qc_{nqubits}_{latent_dim}_{layers}.2dsubhist3", "w")
for b1 in range(0,nbin):
	for b2 in range(0,nbin):
			outf1a.write("%.3e %.3e  %d   %d\n" % (xarr1[b1],yarr1[b2],Rtv1[b1,b2],Rfv1[b1,b2]))
			outf1b.write("%.3e %.3e  %d   %d\n" % (xarr2[b1],yarr2[b2],Rtv2[b1,b2],Rfv2[b1,b2]))
			outf1c.write("%.3e %.3e  %d   %d\n" % (xarr3[b1],yarr3[b2],Rtv3[b1,b2],Rfv3[b1,b2]))
outf1a.close			
outf1b.close			
outf1c.close			

# ### no baseline this time

meas= (np.sum(np.abs(aval-aval2)))/nbin**2

#print(aval-aval2,meas)
print(meas)

outf3 = open("measure.3dhist_${lay}", "a")
outf3.write("%d %d %d %.3f\n" % (nqubits,latent_dim,layers,meas))
outf3.close

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
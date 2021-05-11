#!/bin/bash

#sed -e 's/ /\n/g' histo | sed -e '/^$/d' > target.dat
cat ./out.qgen.target > target.dat
nconf=`awk 'END{print NR}' target.dat`
nfac=1.0
nbin=20


shVAR1="$nconf" python - << MARKER
import numpy as np
import os
from math import sqrt
from math import pi
from random import randint
from random import seed
from scipy import linalg as LA

# ### set parameters

nconf=int(os.environ['shVAR1'])

# ### load data

data = np.loadtxt("target.dat")
outf1 = open("target.hst", "w")

#nbin=int($nconf*$nfac)
nbin=$nbin

########## PERFORM CALCULATION FOR MEAN

histo=[]
for i in range(0,nconf):
	rnd=i 
	histo.append(data[i])

fhist , fedge =np.histogram(histo,bins=nbin,range=(-1,1))
	
for b in range(0,nbin):
	outf1.write("%.14e %d\n" % (fedge[b],fhist[b]))

outf1.close


MARKER

ymax=`sort -n -k 2 target.hst | awk 'END{print $2}'`


###################################################
###################################################
###################################################

cat ./out.qgen.samples > samples.dat
nconf2=`awk 'END{print NR}' samples.dat`

shVAR1="$nconf2" python - << MARKER
import numpy as np
import os
from math import sqrt
from math import pi
from random import randint
from random import seed
from scipy import linalg as LA

# ### set parameters

nconf=int(os.environ['shVAR1'])

# ### load data

data = np.loadtxt("samples.dat")
outf1 = open("samples.hst", "w")

#nbin=int($nconf*$nfac)
nbin=$nbin

########## PERFORM CALCULATION FOR MEAN

histo=[]
for i in range(0,nconf):
	rnd=i 
	histo.append(data[i])

fhist , fedge =np.histogram(histo,bins=nbin,range=(-1,1))
	
for b in range(0,nbin):
	outf1.write("%.14e %d\n" % (fedge[b],fhist[b]))

outf1.close


MARKER

ymax2=`sort -n -k 2 samples.hst | awk 'END{print $2}'`


###################################################
###################################################
###################################################



gnuplot << MARKER

set style fill transparent solid 0.125

###################################################

set lmargin 5
set rmargin 3

set key spacing 1.1
set key at graph(0,0.95), graph(0,0.95)
set key ##font 'Symbol'

set label '{/Symbol r}(data)' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


#set grid
set xr [-1:1]
set yr [0:1.35]
set pointsize 3
set xzeroaxis lt -1
set mytics 4


###########################

m=0
sig=0.25
#a=1/sqrt(2*pi*sig**2)
a=1
f(x) = a * exp( -(x - m)**2 /(2*sig**2) )

plot f(x) lw 5 lc 'red' ti 'target Gaussian' 

###########################

#set style fill solid 0.5 # fill style

repl "target.hst" u (\$1+1./$nbin):(\$2/$ymax) smooth freq w boxes lc rgb "red" ti 'target, sampled'

repl "samples.hst" u (\$1+1./$nbin):(\$2/$ymax2) smooth freq w boxes lc rgb "blue" ti 'generated'


###########################


f2(x) = a2*exp( -(x - m2)**2 /(2*sig2**2) )

fit [x=-0.5:0.5] f2(x) "samples.hst" u (\$1+1./$nbin):(\$2/$ymax2) via a2,m2,sig2

rep f2(x) lw 5 lc 'blue' ti 'fitted Gaussian, x=[-0.5:0.5]' 

###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out 'target.png
rep
set out
set term xterm



MARKER



exit 0

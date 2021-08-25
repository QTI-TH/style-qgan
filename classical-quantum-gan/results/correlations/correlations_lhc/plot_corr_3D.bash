#!/bin/bash



gnuplot << MARKER

set style fill transparent solid 0.125

###################################################

set lmargin 5
set rmargin 2

set key spacing 1.1
set key at graph(0,0.95), graph(0,0.95)
set key ##font 'Symbol'

set title 'histogram D(x,y,z)[real-fake], results relative to baseline'
set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set xr [0.5:6.5]
set yr [0:4]
#set log y

set pointsize 3
set xzeroaxis lt -1
set mytics 4
#set mxtics 4

#set xtics format "%.1E"
set xtics 1

#set ytics format "%.1E"
#set ytics 2e5


###########################

#set style fill solid 0.5 # fill style

plot 1 w l lt 0 dt 2 lw 3 lc 'black' ti 'perfectly matching = 1'

replot 'measure.3dhist_2' u 2:4:5 w yerrorlines ps 3 pt 5 lc rgb "red" ti 'n_{samp}=10k, n_{layers}=2'
replot 'measure.3dhist_4' u 2:4:5 w yerrorlines ps 3 pt 7 lc rgb "blue" ti '5'



###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_mhisto.png
rep
set out
set term xterm



MARKER





gnuplot << MARKER

set style fill transparent solid 0.125

###################################################

set lmargin 7
set rmargin 2

set key spacing 1.1
set key at graph(0,0.95), graph(0,0.85)
set key ##font 'Symbol'

set title 'KL divergences via histogram D(x,y,z) projections'
set label 'KL_{div}(D_{real},D_{fake}),  x,y,z summed ' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set xr [0.5:6.5]
set yr [0.001:10]
set log y
set pointsize 3
set xzeroaxis lt -1
set mytics 4
#set mxtics 4

#set xtics format "%.1E"
set xtics 1

set ytics format "%.0e"
#set ytics 2e5


###########################

#set style fill solid 0.5 # fill style

plot 0.000001 w l lt 0 dt 2 lw 3 lc 'black' ti 'perfectly matching = 0'


replot 'measure.veig2_3D_2' u 2:(\$4+\$5+\$6) w lp ps 3 pt 5 lc rgb "red" ti 'n_{samp}=10k, n_{layers}=2'
replot 'measure.veig2_3D_4' u 2:(\$4+\$5+\$6) w lp ps 3 pt 7 lc rgb "blue" ti '4'



###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density_proj.png
rep
set out
set term xterm



MARKER





gnuplot << MARKER

set style fill transparent solid 0.125

###################################################

set lmargin 7
set rmargin 2

set key spacing 1.1
set key at graph(0,0.95), graph(0,0.85)
set key ##font 'Symbol'

set title 'Summed ratio of target and learned covariance matrix EVs'
set label 'R_{EV}' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set xr [0.5:6.5]
#set yr [0:1.35]
set log y
set pointsize 3
set xzeroaxis lt -1
set mytics 4
#set mxtics 4

#set xtics format "%.1E"
set xtics 1

set ytics format "%.0e"
#set ytics 2e5


###########################

f(x)=1+b*exp(-m*x)
m=2
#a=3
b=10000
fit f(x) 'measure.correig_10' u 2:4 via b,m


###########################

#set style fill solid 0.5 # fill style

plot 1 w l lt 0 dt 2 lw 3 lc 'black' ti 'perfectly matching = 1'

replot 'measure.correig_2' u 2:4 w lp ps 3 pt 5 lc rgb "red" ti 'n_{samp}=10k, n_{layers}=2'
replot 'measure.correig_4' u 2:4 w lp ps 3 pt 7 lc rgb "blue" ti '4'

rep f(x) w l lt -1 lw 3 dt 4 lc 'grey' ti 'f(x)=1+b exp(-mx)'



###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_correig.png
rep
set out
set term xterm



MARKER



gnuplot << MARKER

set style fill transparent solid 0.125

###################################################

set lmargin 7
set rmargin 2

set key spacing 1.1
set key at graph(0,0.95), graph(0,0.85)
set key ##font 'Symbol'

set title 'Summed ratio of target and learned covariance matrices'
set label 'R_{cov}' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set xr [0.5:6.5]
#set yr [0:1.35]
#set log y
set pointsize 3
set xzeroaxis lt -1
set mytics 4
#set mxtics 4

#set xtics format "%.1E"
set xtics 1

set ytics format "%.0e"
#set ytics 2e5


###########################

f(x)=1+b*exp(-m*x)
m=2
#a=3
b=10000
#fit f(x) 'measure.correig_4' u 2:4 via b,m


###########################

#set style fill solid 0.5 # fill style

plot 1 w l lt 0 dt 2 lw 3 lc 'black' ti 'perfectly matching = 1'

replot 'measure.correig_2' u 2:5 w lp ps 3 pt 5 lc rgb "red" ti 'n_{samp}=10k, n_{layers}=2'
replot 'measure.correig_4' u 2:5 w lp ps 3 pt 7 lc rgb "blue" ti '4'

#rep f(x) w l lt -1 lw 3 lc 'black' ti 'f(x)=1+b exp(-mx)'



###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_corrsum.png
rep
set out
set term xterm



MARKER




gnuplot << MARKER

set style fill transparent solid 0.125

###################################################

set lmargin 7
set rmargin 2

set key spacing 1.1
set key at graph(0,0.95), graph(0,0.85)
set key ##font 'Symbol'

set title 'KL divergences from learned covmat'
set label 'KL_{div}(D_{real},D_{fake}), x,y,z summed' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set xr [0.5:6.5]
set yr [1e-6:1e-1]
set log y
set pointsize 3
set xzeroaxis lt -1
set mytics 4
#set mxtics 4

#set xtics format "%.1E"
set xtics 1

set ytics format "%.0e"
#set ytics 2e5


###########################

#set style fill solid 0.5 # fill style

plot 1e-9 w l lt 0 dt 2 lw 3 lc 'black' ti 'perfectly matching = 0'

replot 'measure.corr_2' u 2:7 w lp ps 3 pt 5 lc rgb "red" ti 'n_{samp}=10k, n_{layers}=2'
replot 'measure.corr_5' u 2:7 w lp ps 3 pt 7 lc rgb "blue" ti '4'


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_corrdiv.png
rep
set out
set term xterm



MARKER

exit 0

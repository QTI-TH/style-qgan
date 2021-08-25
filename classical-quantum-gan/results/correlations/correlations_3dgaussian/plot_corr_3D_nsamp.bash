#!/bin/bash

lay=2





VAL='4 5 6 7 8 9'
for val in $VAL
do

awk -v "val=$val" '{print $2,$val}' measure.correig_${lay}_1k > tmp1
awk -v "val=$val" '{print $2,$val}' measure.correig_${lay}_5k > tmp2
awk -v "val=$val" '{print $2,$val}' measure.correig_${lay}_10k > tmp3
awk -v "val=$val" '{print $2,$val}' measure.correig_${lay}_20k > tmp4

join tmp1 tmp2 > tmp12
join tmp3 tmp4 > tmp34
join tmp12 tmp34 | awk '{avg=0.25*($2+$3+$4+$5); dev= sqrt( 0.25*( ($2-avg)**2+($3-avg)**2+($4-avg)**2+($5-avg)**2) ); print $1,avg,dev,dev/avg }' > nsamp_avg_dev.$val


done

join nsamp_avg_dev.4 nsamp_avg_dev.5 | awk '{printf("%d & %.1f & %.1f\n", $1,$4*100,$7*100)}' > tmp12
join nsamp_avg_dev.6 nsamp_avg_dev.7 | awk '{printf("%d & %.1f & %.1f\n", $1,$4*100,$7*100)}' > tmp34
join nsamp_avg_dev.8 nsamp_avg_dev.9 | awk '{printf("%d & %.1f & %.1f\n", $1,$4*100,$7*100)}' > tmp56

join tmp12 tmp34 > tmp1234
join tmp1234 tmp56

rm tmp*

#exit 1


gnuplot << MARKER

set style fill transparent solid 0.125

###################################################

set lmargin 7
set rmargin 2

set key spacing 1.1
set key at graph(0,0.95), graph(0,0.95)
set key ##font 'Symbol'

set title 'Summed ratio of target and learned covariance matrix EVs, n_{layers}=${lay}'
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
set yr [0:]
#set log y
set pointsize 3
set xzeroaxis lt -1
set mytics 4
#set mxtics 4

#set xtics format "%.1E"
set xtics 1

#set ytics format "%.0e"
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

replot 'nsamp_avg_dev.4' u 1:2:3 w yerrorlines ps 3 lw 5 pt 5 lc rgb "red" ti 'n_{samp}=1,5,10,20k, (target,fake)'
replot 'nsamp_avg_dev.5' u 1:2:3 w yerrorlines ps 3 lw 5 pt 7 lc rgb "blue" ti '(target,exact)'
replot 'nsamp_avg_dev.6' u 1:2:3 w yerrorlines ps 3 lw 5 pt 9 lc rgb "orange" ti '(exact,fake)'


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_correig_${lay}_nsamp_err.png
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
set key at graph(0,0.95), graph(0,0.95)
set key ##font 'Symbol'

set title 'Summed ratio of target and learned covariance matrices, n_{layers}=${lay}'
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
set yr [:2.5]
#set log y
set pointsize 3
set xzeroaxis lt -1
set mytics 4
#set mxtics 4

#set xtics format "%.1E"
set xtics 1

#set ytics format "%.0e"
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

replot 'nsamp_avg_dev.7' u 1:2:3 w yerrorlines ps 3 lw 5 pt 5 lc rgb "red" ti 'n_{samp}=1,5,10,20k, (target,fake)'
replot 'nsamp_avg_dev.8' u 1:2:3 w yerrorlines ps 3 lw 5 pt 7 lc rgb "blue" ti '(target,exact)'
replot 'nsamp_avg_dev.9' u 1:2:3 w yerrorlines ps 3 lw 5 pt 9 lc rgb "orange" ti '(exact,fake)'


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_corrsum_${lay}_nsamp_err.png
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
set key at graph(0,0.95), graph(0,0.95)
set key ##font 'Symbol'

set title 'Summed ratio of target and learned covariance matrix EVs, n_{layers}=${lay}'
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
set yr [0:]
#set log y
set pointsize 3
set xzeroaxis lt -1
set mytics 4
#set mxtics 4

#set xtics format "%.1E"
set xtics 1

#set ytics format "%.0e"
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

replot 'measure.correig_${lay}_1k' u 2:4 w lp ps 3 pt 5 lc rgb "red" ti '(target,fake), n_{samp}=1,5,10,20k'
replot 'measure.correig_${lay}_5k' u 2:4 w lp ps 3 pt 5 lc rgb "red" noti
replot 'measure.correig_${lay}_10k' u 2:4 w lp ps 3 pt 5 lc rgb "red" noti
replot 'measure.correig_${lay}_20k' u 2:4 w lp ps 3 pt 5 lc rgb "red" noti

replot 'measure.correig_${lay}_1k' u 2:5 w lp ps 3 pt 7 lc rgb "blue" ti '(target,exact), n_{samp}=1,5,10,20k'
replot 'measure.correig_${lay}_5k' u 2:5 w lp ps 3 pt 7 lc rgb "blue" noti
replot 'measure.correig_${lay}_10k' u 2:5 w lp ps 3 pt 7 lc rgb "blue" noti
replot 'measure.correig_${lay}_20k' u 2:5 w lp ps 3 pt 7 lc rgb "blue" noti

replot 'measure.correig_${lay}_1k' u 2:6 w lp ps 3 pt 9 lc rgb "orange" ti '(exact,fake), n_{samp}=1,5,10,20k'
replot 'measure.correig_${lay}_5k' u 2:6 w lp ps 3 pt 9 lc rgb "orange" noti
replot 'measure.correig_${lay}_10k' u 2:6 w lp ps 3 pt 9 lc rgb "orange" noti
replot 'measure.correig_${lay}_20k' u 2:6 w lp ps 3 pt 9 lc rgb "orange" noti


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_correig_${lay}_nsamp.png
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
set key at graph(0,0.95), graph(0,0.95)
set key ##font 'Symbol'

set title 'Summed ratio of target and learned covariance matrices, n_{layers}=${lay}'
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
set yr [:2.5]
#set log y
set pointsize 3
set xzeroaxis lt -1
set mytics 4
#set mxtics 4

#set xtics format "%.1E"
set xtics 1

#set ytics format "%.0e"
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

replot 'measure.correig_${lay}_1k' u 2:7 w lp ps 3 pt 5 lc rgb "red" ti '(target,fake), n_{samp}=1,5,10,20k'
replot 'measure.correig_${lay}_5k' u 2:7 w lp ps 3 pt 5 lc rgb "red" noti
replot 'measure.correig_${lay}_10k' u 2:7 w lp ps 3 pt 5 lc rgb "red" noti
replot 'measure.correig_${lay}_20k' u 2:7 w lp ps 3 pt 5 lc rgb "red" noti

replot 'measure.correig_${lay}_1k' u 2:8 w lp ps 3 pt 7 lc rgb "blue" ti '(target,exact), n_{samp}=1,5,10,20k'
replot 'measure.correig_${lay}_5k' u 2:8 w lp ps 3 pt 7 lc rgb "blue" noti
replot 'measure.correig_${lay}_10k' u 2:8 w lp ps 3 pt 7 lc rgb "blue" noti
replot 'measure.correig_${lay}_20k' u 2:8 w lp ps 3 pt 7 lc rgb "blue" noti

replot 'measure.correig_${lay}_1k' u 2:9 w lp ps 3 pt 9 lc rgb "orange" ti '(exact,fake), n_{samp}=1,5,10,20k'
replot 'measure.correig_${lay}_5k' u 2:9 w lp ps 3 pt 9 lc rgb "orange" noti
replot 'measure.correig_${lay}_10k' u 2:9 w lp ps 3 pt 9 lc rgb "orange" noti
replot 'measure.correig_${lay}_20k' u 2:9 w lp ps 3 pt 9 lc rgb "orange" noti


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_corrsum_${lay}_nsamp.png
rep
set out
set term xterm



MARKER
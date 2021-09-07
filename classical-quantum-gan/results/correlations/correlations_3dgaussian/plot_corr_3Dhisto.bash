#!/bin/bash



gnuplot << MARKER

#set style fill transparent solid 0.125

###################################################

set lmargin 2
set rmargin 2
set bmargin 1
set tmargin 1


set key spacing 1.1
set key at graph(0,0.95), graph(0,0.95)
set key ##font 'Symbol'

set title 'D(x,y,z)[real], (y,z) projected'
#set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
#set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


#set grid
set xr [-1:1]
set yr [-1:1]
set zr [0:]
#set log y

set pointsize 3
#set xzeroaxis lt -1
#set mytics 4
#set mxtics 4

#set xtics format "%.1E"
set ztics 40

#set ytics format "%.1E"
#set ytics 2e5

###########################


#set view map
set pm3d at s 
set dgrid3d 40,40
#set pm3d interpolate 1,1 

#set palette rgbformulae 33,13,10 
set palette defined (0 "white", 0.25 "orange", 1 "red")

set view 30
###########################

#set style fill solid 0.5 # fill style

splot 'results/qc_3_1_2.2dsubhist' using 1:2:4 w l lw 0 lc 'black' noti 


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density1_yz_target.png
rep
set out
set term xterm



MARKER

exit 


LAT='1 2 3 4 5 6'
LAY='2 5 7 10'
#LAT='1'
#LAY='2'
for lat in $LAT
do
	for lay in $LAY
	do







gnuplot << MARKER

#set style fill transparent solid 0.125

###################################################

set lmargin 2
set rmargin 2
set bmargin 1
set tmargin 1


set key spacing 1.1
set key at graph(0,0.95), graph(0,0.95)
set key ##font 'Symbol'

set title 'D(x,y,z)[real] - D(x,y,z)[fake], n_{layers}=${lay}, d_{latent}=${lat}'
#set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
#set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set xr [-1:1]
set yr [-1:1]
set zr [-10:20]
#set log y

set pointsize 3
#set xzeroaxis lt -1
#set mytics 4
#set mxtics 4

#set xtics format "%.1E"
#set xtics 1

#set ytics format "%.1E"
#set ytics 2e5

###########################


#set view map
set pm3d #at b 
set dgrid3d 40,40
#set pm3d interpolate 1,1

set palette defined (-10 "blue", 0 "white", 10 "orange", 20 "red")

set view 40

###########################

offset=0 #-4.8

#set style fill solid 0.5 # fill style

splot 'results/qc_3_${lat}_${lay}.2dsubhist' using 1:2:(\$7-\$4+offset) w l lw 1 lc 'black' noti 


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density1_yz_${lat}_${lay}.png
rep
set out
set term xterm



MARKER


done
done



exit 0

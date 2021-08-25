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

set title 'D(x,y,z)[real], (E^2,M_T) '
#set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
#set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


#set grid
set xr [0:9e5]
set yr [-6e5:0]
#set zr [0:]
#set log y

set pointsize 3
#set xzeroaxis lt -1
#set mytics 4
#set mxtics 4

set xtics format "%.0e"
set ztics 100

set ytics format "%.0e"
set xtics 2e5
set ytics 2e5

###########################

set colorbox user origin 0.87,0.55 size 0.03,0.4
#set view map
set pm3d at s 
set dgrid3d 15,15
#set pm3d interpolate 1,1 

#set palette rgbformulae 33,13,10 
set palette defined (0 "white", 0.25 "orange", 1 "red")

set view 30
###########################

#set style fill solid 0.5 # fill style

splot 'results/qc_3_5_2.aws.2dsubhist1' using 1:2:3 w l lw 0 lc 'black' noti 


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density1_EM_target_aws.png
rep
set out
set term xterm



MARKER

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

set title 'D(x,y,z)[real], (E^2,R) '
#set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
#set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


#set grid
set xr [0:9e5]
set yr [-4:4]
#set zr [0:]
#set log y

set pointsize 3
#set xzeroaxis lt -1
#set mytics 4
#set mxtics 4

set xtics format "%.0e"
set ztics 100

#set ytics format "%.0e"
set xtics 2e5
#set ytics 2e5

###########################

set colorbox user origin 0.87,0.55 size 0.03,0.4
#set view map
set pm3d at s 
set dgrid3d 15,15
#set pm3d interpolate 1,1 

#set palette rgbformulae 33,13,10 
set palette defined (0 "white", 0.25 "orange", 1 "red")

set view 30
###########################

#set style fill solid 0.5 # fill style

splot 'results/qc_3_5_2.aws.2dsubhist2' using 1:2:3 w l lw 0 lc 'black' noti 


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density1_ER_target_aws.png
rep
set out
set term xterm



MARKER



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

set title 'D(x,y,z)[real], (M_T,R) '
#set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
#set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


#set grid
set yr [-4:4]
set xr [-6e5:0]
#set zr [0:]
#set log y

set pointsize 3
#set xzeroaxis lt -1
#set mytics 4
#set mxtics 4

set xtics format "%.0e"
set ztics 100

#set ytics format "%.0e"
set xtics 2e5
#set ytics 2e5

###########################

set colorbox user origin 0.87,0.55 size 0.03,0.4
#set view map
set pm3d at s 
set dgrid3d 15,15
#set pm3d interpolate 1,1 

#set palette rgbformulae 33,13,10 
set palette defined (0 "white", 0.25 "orange", 1 "red")

set view 30
###########################

#set style fill solid 0.5 # fill style

splot 'results/qc_3_5_2.aws.2dsubhist3' using 1:2:3 w l lw 0 lc 'black' noti 


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density1_MR_target_aws.png
rep
set out
set term xterm



MARKER

#exit 1



LAT='5'
LAY='2'
LAB='aws aws2'

for lat in $LAT
do
	for lay in $LAY
	do

### real-fake ################################################################################################################
##############################################################################################################################

		for lab in $LAB
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

set title 'D(E^2,M_T)[real] - D(E^2,M_T)[fake], n_{layers}=${lay}, d_{latent}=${lat}'
#set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
#set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set xr [0:9e5]
set yr [-6e5:0]
#set zr [-10:20]
#set log y

set pointsize 3
#set xzeroaxis lt -1
#set mytics 4
#set mxtics 4

set xtics format "%.0e"
set ztics 30

set ytics format "%.0e"
set xtics 2e5
set ytics 2e5

###########################


set colorbox user origin 0.87,0.55 size 0.03,0.4
#set view map
set pm3d #at b 
set dgrid3d 15,15
#set pm3d interpolate 1,1

set palette defined (-50 "blue", 0 "white", 20 "orange", 50 "red")

set view 30

###########################

offset=0 #-4.8

#set style fill solid 0.5 # fill style

splot 'results/qc_3_${lat}_${lay}.${lab}.2dsubhist1' using 1:2:(\$3-\$4) w l lw 1 lc 'black' noti 


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density1_EM_${lat}_${lay}_${lab}.png
rep
set out
set term xterm



MARKER


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

set title 'D(E^2,R)[real] - D(E^2,R)[fake], n_{layers}=${lay}, d_{latent}=${lat}'
#set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
#set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set xr [0:9e5]
set yr [-4:4]
#set zr [-10:20]
#set log y

set pointsize 3
#set xzeroaxis lt -1
#set mytics 4
#set mxtics 4

set xtics format "%.0e"
set ztics 30

#set ytics format "%.0e"
set xtics 2e5
#set ytics 2e5

###########################


set colorbox user origin 0.87,0.55 size 0.03,0.4
#set view map
set pm3d #at b 
set dgrid3d 15,15
#set pm3d interpolate 1,1

set palette defined (-30 "blue", 0 "white", 10 "orange", 20 "red")

set view 30

###########################

offset=0 #-4.8

#set style fill solid 0.5 # fill style

splot 'results/qc_3_${lat}_${lay}.${lab}.2dsubhist2' using 1:2:(\$3-\$4) w l lw 1 lc 'black' noti 


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density1_ER_${lat}_${lay}_${lab}.png
rep
set out
set term xterm



MARKER



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

set title 'D(E^2,M_T)[real] - D(E^2,M_T)[fake], n_{layers}=${lay}, d_{latent}=${lat}'
#set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
#set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set yr [-4:4]
set xr [-6e5:0]
#set zr [-10:20]
#set log y

set pointsize 3
#set xzeroaxis lt -1
#set mytics 4
#set mxtics 4

set xtics format "%.0e"
set ztics 30

#set ytics format "%.0e"
set xtics 2e5
#set ytics 2e5

###########################


set colorbox user origin 0.87,0.55 size 0.03,0.4
#set view map
set pm3d #at b 
set dgrid3d 15,15
#set pm3d interpolate 1,1

set palette defined (-25 "blue", 0 "white", 15 "orange", 30 "red")

set view 30

###########################

offset=0 #-4.8

#set style fill solid 0.5 # fill style

splot 'results/qc_3_${lat}_${lay}.${lab}.2dsubhist3' using 1:2:(\$3-\$4) w l lw 1 lc 'black' noti 


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density1_MR_${lat}_${lay}_${lab}.png
rep
set out
set term xterm



MARKER

done





### simulated - fake #########################################################################################################
##############################################################################################################################

		



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

set title 'D(E^2,M_T)[fake] - D(E^2,M_T)[fake], n_{layers}=${lay}, d_{latent}=${lat}'
#set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
#set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set xr [0:9e5]
set yr [-6e5:0]
#set zr [-10:20]
#set log y

set pointsize 3
#set xzeroaxis lt -1
#set mytics 4
#set mxtics 4

set xtics format "%.0e"
set ztics 30

set ytics format "%.0e"
set xtics 2e5
set ytics 2e5

###########################


set colorbox user origin 0.87,0.55 size 0.03,0.4
#set view map
set pm3d #at b 
set dgrid3d 15,15
#set pm3d interpolate 1,1

set palette defined (-50 "blue", 0 "white", 20 "orange", 50 "red")

set view 30

###########################

offset=0 #-4.8

#set style fill solid 0.5 # fill style

splot 'results/qc_3_${lat}_${lay}.aws2.2dsubhist1' using 1:2:(\$4-\$5) w l lw 1 lc 'black' noti 


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density1_EM_${lat}_${lay}_fakes.png
rep
set out
set term xterm



MARKER


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

set title 'D(E^2,R)[fake] - D(E^2,R)[fake], n_{layers}=${lay}, d_{latent}=${lat}'
#set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
#set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set xr [0:9e5]
set yr [-4:4]
#set zr [-10:20]
#set log y

set pointsize 3
#set xzeroaxis lt -1
#set mytics 4
#set mxtics 4

set xtics format "%.0e"
set ztics 30

#set ytics format "%.0e"
set xtics 2e5
#set ytics 2e5

###########################


set colorbox user origin 0.87,0.55 size 0.03,0.4
#set view map
set pm3d #at b 
set dgrid3d 15,15
#set pm3d interpolate 1,1

set palette defined (-30 "blue", 0 "white", 10 "orange", 20 "red")

set view 30

###########################

offset=0 #-4.8

#set style fill solid 0.5 # fill style

splot 'results/qc_3_${lat}_${lay}.aws2.2dsubhist2' using 1:2:(\$4-\$5) w l lw 1 lc 'black' noti 


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density1_ER_${lat}_${lay}_fakes.png
rep
set out
set term xterm



MARKER



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

set title 'D(E^2,M_T)[fake] - D(E^2,M_T)[fake], n_{layers}=${lay}, d_{latent}=${lat}'
#set label 'histogram overlap, nbin=40' at graph(0,0.025), graph(0,0.92) ##font 'Symbol'
#set label 'd_{latent}' at graph(0,0.88), graph(0,0.075) ##font 'Symbol'

#LABEL="Very preliminary"
#set obj 10 rect at graph(0,0.82), graph(0,0.875) size char strlen(LABEL), char 2 
#set obj 10 fs solid 0.5 fc "pink" behind
#set label 'Very preliminary' at graph(0,0.7), graph(0,0.87)

#set arrow from 1000,graph(0,0) to 1000, graph(1,1) nohead lt 0 lw 2
set arrow from 192,graph(0,0) to 192, graph(0.4,0.4) nohead lt 1 lw 2 lc black


set grid
set yr [-4:4]
set xr [-6e5:0]
#set zr [-10:20]
#set log y

set pointsize 3
#set xzeroaxis lt -1
#set mytics 4
#set mxtics 4

set xtics format "%.0e"
set ztics 30

#set ytics format "%.0e"
set xtics 2e5
#set ytics 2e5

###########################


set colorbox user origin 0.87,0.55 size 0.03,0.4
#set view map
set pm3d #at b 
set dgrid3d 15,15
#set pm3d interpolate 1,1

set palette defined (-25 "blue", 0 "white", 15 "orange", 30 "red")

set view 30

###########################

offset=0 #-4.8

#set style fill solid 0.5 # fill style

splot 'results/qc_3_${lat}_${lay}.aws2.2dsubhist3' using 1:2:(\$4-\$5) w l lw 1 lc 'black' noti 


###########################



set term pngcairo size 1600, 1200 truecolor enhanced font "Helvetica,34"
set out '3dgauss_density1_MR_${lat}_${lay}_fakes.png
rep
set out
set term xterm



MARKER



done
done



exit 0

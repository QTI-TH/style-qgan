#!/bin/bash

LAY='2 5 7 10'
LAT='1 2 3 4 5 6'
for lay in $LAY
do
for lat in $LAT
do

python3.8 plots_3dgaussian.py --latent_dim=$lat --layers=$lay

done
done

exit 0
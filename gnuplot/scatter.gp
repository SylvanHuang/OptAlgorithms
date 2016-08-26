
# This is a small script to draw the histograms with the population.dat

set terminal epslatex color solid standalone 'ptm' 14 \
    header '\usepackage{xcolor, amsmath}'

set output 'scatter.tex'

set macros

set key off


set xlabel '$x_1$'
set xtics 0.002
set ylabel '$x_2$' offset 2
set ytics 0.002

#set style fill solid 1.0 noborder

plot 'population.dat' using (exp($1)):(exp($2)) with points pointtype 6 pointsize 1.5

set output


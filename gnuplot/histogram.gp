
# This is a small script to draw the histograms with the population.dat

set terminal epslatex color solid standalone 'ptm' 14 \
    header '\usepackage{xcolor, amsmath}'

set output 'histogram.tex'

set macros

set key off

set border 3

set xlabel '$x_1$'
#set xlabel '$x_2$'
set xtics 0.002

set ylabel 'Frequency'
set ytics 300

set boxwidth 0.0004 absolute
#set boxwidth 0.00002 absolute

set style fill solid 1.0 noborder

bin_width=0.0005
#bin_width=0.000025

bin_number(x)=floor(x/bin_width)

rounded(x)=bin_width * (bin_number(x))

plot 'population.dat' using (rounded($2) ):(1) smooth frequency with boxes

set output


#!/usr/bin/env bash

# first argument: input directory
# second argument: output directory for ThermoRawFileParser

# convert .RAW to mzML
cd ~
cd tools/ThermoRawFileParser
OUTDIR="$(dirname "$2")"
mkdir -p $2
mono ThermoRawFileParser.exe -d=$1 -o=$2 -p=1 -f=1 -m=1 -L=1 -l=2 # L=1, MS level1; p=1, no peakpicking for profile data; f=1, file format mzML; m=1, metadata in txt format; l logging default

# msconvert to ProtMSD input
mkdir -p $OUTDIR/msconvert
mzmls=`ls $2/*.mzML`
for eachfile in $mzmls
do
   msconvert $eachfile -o $OUTDIR/msconvert
done

#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/cmnfs/data/protMSD/slurm-%j.out
#SBATCH --job-name=pMSDpp
#SBATCH --mem-per-cpu=100
#SBATCH --mail-user=zixuan.xiao@tum.de

# arg1: workflow type: mascot or maxquant
# arg2: mzML file
# arg3: maxquant or mascot file
# arg4: mascot reference file

# get configs
. /cmnfs/proj/ORIGINS/protMSD/maxquant/pipeline_config.sh

# protMSD
echo "================== Load configurations =================="
source ~/proj_env/py27/py27/bin/activate 

# Create relevant folder
dirname=$(dirname "$2")
basename=$(basename "$2")
filename="${basename%.*}_rtgw${rtgw}_n${n_noise}_mm${mm}_isospec${isospec}_dev${dev}_initWithCos${initWithCos}_IDwithCos${IDwithCos}"
echo "Filename: $filename"

# if [ ! -d "$dirname/$filename" ]; then
#     mkdir -p "$dirname/$filename"
#     echo "Create protMSD result directory $dirname/$filename"
# fi

# if [ "$1" = "maxquant" ]; then
#     echo " ==================ProtMSD pipeline with $1================== "
#     python -u /cmnfs/proj/ORIGINS/protMSD/maxquant/protNMF/maxquant_handler.py -ml $2 -MQ $3 -p $p -rtgw $rtgw -n_noise $n_noise -mm $mm
# elif [ "$1" = "mascot" ]; then
#     echo " ==================ProtMSD pipeline with $1================== "
#     python -u /cmnfs/proj/ORIGINS/protMSD/maxquant/protNMF/mascot_handler.py -ml $2 -MQ $3 -p $p -rtgw $rtgw -n_noise $n_noise -mm $mm
# fi
# deactivate

# analyze result
echo "================== Result analysis =================="
if [[ "$1" = "mascot" ]]; then
    ref_fname="$4"
    echo "reference file: $ref_fname"
    conda run -n py310 python -u /cmnfs/proj/ORIGINS/protMSD/maxquant/analysis/result_analysis.py -dir "$dirname/$filename" -ref_type $1 \
    -fname $filename -int_thres $int_thres -ref_fname $ref_fname #avoid conda activate in shell script: it is for interactive session
else
    conda run -n py310 python -u /cmnfs/proj/ORIGINS/protMSD/maxquant/analysis/result_analysis.py -dir "$dirname/$filename" -ref_type $1 \
    -fname $filename -int_thres $int_thres
fi




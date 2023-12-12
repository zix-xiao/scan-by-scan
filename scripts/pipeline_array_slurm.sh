#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/cmnfs/data/protMSD/slurm-%A-%a.out
#SBATCH --job-name=pMSDpp_arr
#SBATCH --mem-per-cpu=100
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=zixuan.xiao@tum.de
#SBATCH --array=1-30%10

# arg1: workflow type: mascot or maxquant
# arg2: mzML file
# arg3: maxquant or mascot file
# arg4: mascot reference file

# get configs
echo "================== Load configurations =================="
. /cmnfs/proj/ORIGINS/protMSD/maxquant/pipeline_config.sh

config=/cmnfs/proj/ORIGINS/protMSD/maxquant/analysis/arrray_config.txt
isospec=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
mm=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
rtgw=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

echo "This is array task ${SLURM_ARRAY_TASK_ID}, IsoSpec usage is ${isospec}, with bin size 1 over ${mm} and gaussian peak width of ${rtgw}."

# protMSD

source ~/proj_env/py27/py27/bin/activate 

# Create relevant folder
dirname=$(dirname "$2")
basename=$(basename "$2")
filename="${basename%.*}_rtgw${rtgw}_n${n_noise}_mm${mm}"
if [[ $isospec == "True" ]]; then
    filename+='_isospec'
fi
echo "Filename: $filename"

if [ ! -d "$dirname/$filename" ]; then
    mkdir -p "$dirname/$filename"
    echo "Create protMSD result directory $dirname/$filename"
fi

if [ "$1" = "maxquant" ]; then
    echo " ==================ProtMSD pipeline with $1================== "
    if [[ $isospec == "True" ]]; then
        python -u /cmnfs/proj/ORIGINS/protMSD/maxquant/protNMF/maxquant_handler.py -ml $2 -MQ $3 -p $p -rtgw $rtgw -n_noise $n_noise -mm $mm -isospec $isospec
    else
        python -u /cmnfs/proj/ORIGINS/protMSD/maxquant/protNMF/maxquant_handler.py -ml $2 -MQ $3 -p $p -rtgw $rtgw -n_noise $n_noise -mm $mm # by default no isospec
    fi
elif [ "$1" = "mascot" ]; then
    echo " ==================ProtMSD pipeline with $1================== "
    python -u /cmnfs/proj/ORIGINS/protMSD/maxquant/protNMF/mascot_handler.py -ml $2 -MQ $3 -p $p -rtgw $rtgw -n_noise $n_noise -mm $mm
fi
deactivate

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






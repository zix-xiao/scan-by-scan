#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/cmnfs/proj/ORIGINS/data/protMSD/SBS_RTalpha_NoIntercept-%A-%a.out
#SBATCH --job-name=SBS
#SBATCH --mem-per-cpu=400
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=zixuan.xiao@tum.de
#SBATCH --array=11-15

mzML='/cmnfs/proj/ORIGINS/data/protMSD/GCF_profile/msconvert_profile.mzML'
maxquant='/cmnfs/proj/ORIGINS/data/protMSD/GCF_profile/combined/txt/evidence.txt'
method='peakRange'

config=/cmnfs/proj/ORIGINS/protMSD/maxquant/array_scanbyscan_rt_alpha.txt # ToChange
RT_tol=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
alpha=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

echo "This is array task ${SLURM_ARRAY_TASK_ID}, with retention time tolerance ${RT_tol} minutes and mz value tolerance of ${mz_tol}."
## printing shell variables is complicated by escaping
conda run -n py310 python -u /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/scripts/ScanByScan.py -ml $mzML -MQ $maxquant -RT_tol $RT_tol -cond $method -alpha $alpha
echo "Scan by Scan optimization finished."




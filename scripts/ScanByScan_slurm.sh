#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/cmnfs/proj/ORIGINS/data/protMSD/slurm_out/%x_%j.out
#SBATCH --job-name=SBS
#SBATCH --mem-per-cpu=16Gb
#SBATCH --mail-user=zixuan.xiao@tum.de


config_path='/cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/utils/sbs_config.json'

source $HOME/condaInit.sh
conda activate sbs
python /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/sbs_runner.py \
--config_path=$config_path || exit 91

echo "Script finished"
exit 0
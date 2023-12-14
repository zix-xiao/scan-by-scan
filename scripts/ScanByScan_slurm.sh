#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/cmnfs/proj/ORIGINS/data/protMSD/slurm_out/%x_%j.out
#SBATCH --job-name=SBS
#SBATCH --mem-per-cpu=50
#SBATCH --mail-user=zixuan.xiao@tum.de


#mzML='/cmnfs/proj/ORIGINS/data/protMSD/GCF_profile/msconvert_profile.mzML'
mzML='/cmnfs/proj/ORIGINS/data/ecoli/ss/DDA/raw/msconvert/BBM_647_P241_02_07_ssDDA_MIA_001.mzML'
maxquant='/cmnfs/proj/ORIGINS/data/ecoli/HpHRP/MQ/1FDR/combined/txt/evidence_1_FilteredByClosestRT_transfer_RT_pred_filtered_withIso_expRT.pkl'
maxquant_exp='/cmnfs/proj/ORIGINS/data/ecoli/ss/DDA/MQ/combined/txt/evidence_1.txt'
#maxquant='/cmnfs/proj/ORIGINS/data/protMSD/GCF_profile/combined/txt/evidence_transfer_RT_pred_filtered_withIso.pkl'

## printing shell variables is complicated by escaping
source $HOME/condaInit.sh
conda activate sbs
python /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/ScanByScan.py \
--mzml_path=$mzML \
--MQ_ref_path=$maxquant \
--RT_tol=1.0 \
--opt-algo='lasso_cd' \
--RT_ref='pred' \
--MQ_exp_path=$maxquant_exp \
--notes='1FDR_' \
--PS_cos_dist='True' || exit 91

echo "Script finished"
exit 0
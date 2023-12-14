#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/cmnfs/proj/ORIGINS/data/protMSD/slurm_out/%x_%j.out
#SBATCH --job-name=extract_peak
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=50
#SBATCH --mail-user=zixuan.xiao@tum.de

MQ_ref_path=/cmnfs/proj/ORIGINS/data/ecoli/HpHRP/MQ/1FDR/combined/txt/evidence_1_FilteredByClosestRT_transfer_RT_pred_filtered_withIso_expRTrange.pkl
MQ_exp_path=/cmnfs/proj/ORIGINS/data/ecoli/ss/DDA/MQ/combined/txt/evidence_1.txt 
activation_path=/cmnfs/proj/ORIGINS/data/ecoli/ss/DDA/raw/msconvert/BBM_647_P241_02_07_ssDDA_MIA_001_ScanByScan_RTtol1.0_MZtol0.0_condpeakRange_alpha0.0_threshold_abthres0.001_missabthres0.5_convergence_NoIntercept_mix/BBM_647_P241_02_07_ssDDA_MIA_001_ScanByScan_RTtol1.0_MZtol0.0_condpeakRange_alpha0.0_threshold_abthres0.001_missabthres0.5_convergence_NoIntercept_mix_output_activationMinima.npy 
MS1Scans_path="/cmnfs/proj/ORIGINS/data/ecoli/ss/DDA/raw/msconvert/BBM_647_P241_02_07_ssDDA_MIA_001_MS1Scans_NoArray.csv"
source $HOME/condaInit.sh
conda activate sbs

python3 /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/extract_peak.py \
--MQ_ref_path=$MQ_ref_path \
--MQ_exp_path=$MQ_exp_path \
--MS1ScansNoArray_path=$MS1Scans_path \
--activation_path=$activation_path \
--peak_width_thres="(4,None)" \
--ref_RT_apex='Retention time new' \
--ref_RT_start='Retention time start' \
--ref_RT_end='Retention time end' || exit 91

echo "Script finished"
exit 0

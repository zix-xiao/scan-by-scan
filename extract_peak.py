import os
import logging
import time
from unittest import result

import fire
import numpy as np
import pandas as pd

from postprocessing.post_processing import calculate_sum_activation_array
from result_analysis.result_analysis import SBSResult


def extract_peaks_and_analyze(
    MQ_ref_path: str,
    MQ_exp_path: str,
    activation_path: str = (
        "/mnt/cmnfs/proj/ORIGINS/data/ecoli/ss/DDA/raw/msconvert/"
        "1FDR_BBM_647_P241_02_07_ssDDA_MIA_001_ScanByScan_RTtol1.0_MZtolNone_peakRange_lasso_cd_abthres0.001_missabthres0.5_convergence_NoIntercept_pred/"
        "1FDR_BBM_647_P241_02_07_ssDDA_MIA_001_ScanByScan_RTtol1.0_MZtolNone_peakRange_threshold_abthres0.001_missabthres0.5_convergence_NoIntercept_pred_output_activationMinima.npy"
    ),
    MS1ScansNoArray_path: str = (
        "/mnt/cmnfs/proj/ORIGINS/data/ecoli/ss/DDA/raw/msconvert/"
        "BBM_647_P241_02_07_ssDDA_MIA_001_MS1Scans_NoArray.csv"
    ),
    RT_ref: str = "predicted_RT",
    return_peak_result: bool = True,
    peak_width_thres=(3, None),
):
    start_time_init = time.time()
    result_dir = os.path.dirname(activation_path)
    Maxquant_result_dict = pd.read_pickle(filepath_or_buffer=MQ_ref_path)
    Maxquant_result_exp = pd.read_csv(MQ_exp_path, sep="\t")
    MS1Scans_NoArray = pd.read_csv(os.path.join(MS1ScansNoArray_path))
    activation = np.load(activation_path)
    find_peak_cond = "_".join(str(a) for a in peak_width_thres)
    find_peak_cond += "_"+RT_ref.replace(' ', '_')

    logging.basicConfig(
        filename=os.path.join(result_dir, find_peak_cond + ".log"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    if return_peak_result:
        sum_peak, peak_results = calculate_sum_activation_array(
            Maxquant_result=Maxquant_result_dict,  # 100 rows for testing
            activation=activation,
            MS1ScansNoArray=MS1Scans_NoArray,
            RT_ref=RT_ref,
            return_peak_result=return_peak_result,
            peak_width_thres=peak_width_thres,
        )
        sum_peak.to_csv(
            os.path.join(result_dir, "sum_peak_width" + find_peak_cond + ".csv"),
            index=False,
        )
        peak_results.to_csv(
            os.path.join(result_dir, "peak_results_width" + find_peak_cond + ".csv"),
            index=False,
        )
    else:
        sum_peak = calculate_sum_activation_array(
            Maxquant_result=Maxquant_result_dict,
            activation=activation,
            MS1ScansNoArray=MS1Scans_NoArray,
            RT_ref=RT_ref,
            return_peak_result=return_peak_result,
            peak_width_thres=peak_width_thres,
        )
        sum_peak.to_csv(
            os.path.join(result_dir, "sum_peak_width" + find_peak_cond + ".csv"),
            index=False,
        )
    minutes, seconds = divmod(time.time() - start_time_init, 60)
    logging.info("Script execution time: {}m {}s".format(int(minutes), int(seconds)))

    SBS_result = SBSResult(
        ref_df=Maxquant_result_dict,
        exp_df=Maxquant_result_exp,
        RT_tol=1.0,
        sum_peak=sum_peak,
    )
    SBS_result.plot_intensity_corr(
        ref_col="Intensity",
        inf_col="AUCActivationPeak",
        interactive=False,
        save_dir=result_dir,
    )
    minutes, seconds = divmod(time.time() - start_time_init, 60)
    logging.info("Script execution time: {}m {}s".format(int(minutes), int(seconds)))


if __name__ == "__main__":
    fire.Fire(extract_peaks_and_analyze)

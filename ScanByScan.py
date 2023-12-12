import argparse
import logging
import os
import pickle
import time
from multiprocessing import cpu_count
from typing import List, Literal, Union

import fire
import numpy as np
import pandas as pd

import postprocessing.post_processing as post_processing
from optimization.dictionary import LoadMZML, _AlignMethods
from optimization.inference import _algo, _alpha_criteria, process_scans_parallel
from result_analysis import result_analysis

os.environ["NUMEXPR_MAX_THREADS"] = "32"


def opt_scan_by_scan(
    mzml_path: str,
    MQ_ref_path: str,
    RT_tol: float,
    mz_align_method: _AlignMethods = "peakRange",
    mz_tol: float | None = None,
    alpha_criteria: _alpha_criteria = "convergence",
    alphas: Union[List, np.ndarray] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    opt_algo: _algo = "lasso_cd",
    iso_resolution: float = 0.001,
    iso_ab_mis_thres: float = 0.5,
    RT_ref: Literal["pred", "exp", "mix"] = "pred",
    MQ_exp_path: str | None = None,
    notes: str = "",
    PS_cos_dist: bool = False,
):
    """ Scan by scan optimization for joint precursor identification and quantification

    Args:
        mzml_path (str): path to the input data in mzML format, only MS1 level required
        MQ_ref_path (str): path to MaxQuant results (evidence.txt) of the same RAW file, \
            used for constructing reference dictionary",
        RT_tol (float): Tolerance when selecting candidate precursor by retention time, \
            unit is minute
        mz_align_method (_AlignMethods, optional): The method for doing mz alignment. Defaults to 'peakRange'.
        mz_tol (float | None, optional): Tolerance when matching theoratical isotope mz value to observed mz value. \
            Defaults to None.
        alpha_criteria (_alpha_criteria, optional): Criteria for choosing the best alpha, either min or convergence. \
            Defaults to 'convergence'.
        alphas (Union[List, np.ndarray], optional): (TODO) Candidate penalty strength for Lasso regression, \
            shrinking threshold for threshold method. Defaults to [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100].
        opt_algo (_algo, optional): Algorithm to use for sparse encoding. Defaults to 'lasso_cd'.
        iso_resolution (float, optional): Abundance threshold for generating Isotope envelops \
            when constructing dictionary. Defaults to 0.001.
        iso_ab_mis_thres (float, optional): Threshold for maximum allowed missing isotope abundance for a precursor \
            to be included into candidate list. Defaults to 0.5.
        RT_ref (Literal['pred', 'exp', 'mix'], optional):Reference retention time, \
            exp - experimental retention time (from MQ), pred - prediction from DeepLC, \
                mix - exp RT from MQ if available, else from DeepLC. Defaults to 'pred'.
        MQ_exp_path (str | None, optional): path to MaxQuant experiment \
            results (evidence.txt) of the same RAW file, used for comparing inferred intensity. Defaults to None.
        notes (str, optional): extra part that will be appended in the beginning of the name\
              in file folder. Defaults to empty string ''.
    """

    # define paths
    dirname = os.path.dirname(mzml_path)
    basename = os.path.basename(mzml_path)
    MS1Scans_NoArray_name = basename[:-5] + "_MS1Scans_NoArray.csv"
    filename = (
        notes
        + basename[:-5]
        + "_ScanByScan"
        + "_RTtol"
        + str(RT_tol)
        + "_MZtol"
        + str(mz_tol)
        + "_"
        + mz_align_method
        + "_"
        + opt_algo
        + "_abthres"
        + str(iso_resolution)
        + "_missabthres"
        + str(iso_ab_mis_thres)
        + "_"
        + alpha_criteria
        + "_NoIntercept"
        + "_"
        + RT_ref
    )
    if PS_cos_dist:
        filename += "_PScosDist"

    if not os.path.exists(os.path.join(dirname, filename)):
        os.makedirs(os.path.join(dirname, filename))
    filename_full = os.path.join(dirname, filename)
    output_file = os.path.join(filename_full, filename + "_output")  # filename

    result_dir = filename_full
    report_dir = os.path.join(result_dir, "report")
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        os.makedirs(os.path.join(report_dir, "activation"))

    logging.basicConfig(
        filename=os.path.join(filename_full, filename + ".log"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info(
        "Args: msconvert filepath - %s, maxquant result path - %s.",
        mzml_path,
        MQ_ref_path,
    )
    logging.info(
        "Args: RT tolerance - %s, mz tolerance(not relevant) - %s.", RT_tol, mz_tol
    )
    logging.info(
        "Args: mz align method - %s, alpha - %s, isotope abundance threshold - %s.",
        mz_align_method,
        alphas,
        iso_resolution,
    )
    logging.info(
        "Args: maximum missing isotope abundance - %s, alpha criteria - %s.",
        iso_ab_mis_thres,
        alpha_criteria,
    )
    logging.info("Output directory - %s", filename_full)

    # start analysis
    start_time_init = time.time()
    logging.info("==================Load data==================")

    # Load reference data
    # Maxquant_result = pd.read_csv(filepath_or_buffer=maxquant_file, sep='\t')
    Maxquant_result_dict = pd.read_pickle(filepath_or_buffer=MQ_ref_path)
    Maxquant_result_exp = pd.read_csv(filepath_or_buffer=MQ_exp_path, sep="\t")

    # Load MS1 scans from pkl or mzml file
    try:
        with open(mzml_path[:-5] + ".pkl", "rb") as f:
            MS1Scans = pickle.load(f)
        logging.info("Load pre-stored pickle results")
        if not os.path.isfile(os.path.join(dirname, MS1Scans_NoArray_name)):
            MS1Scans_NoArray = MS1Scans.iloc[:, 1:5].copy()
            MS1Scans_NoArray.to_csv(
                os.path.join(dirname, MS1Scans_NoArray_name), index=0
            )
    except:
        logging.info("Pickle result not found, load mzml file.")
        MS1Scans = LoadMZML(mzml_path)
        MS1Scans.to_pickle(mzml_path[:-5] + ".pkl")
    minutes, seconds = divmod(time.time() - start_time_init, 60)
    logging.info("Script execution time: {}m {}s".format(int(minutes), int(seconds)))

    # Set left and right edge for scan search range
    if RT_ref == "exp":
        Maxquant_result_dict["RT_search_left"] = (
            Maxquant_result_dict["Calibrated retention time start"] - RT_tol
        )
        Maxquant_result_dict["RT_search_right"] = (
            Maxquant_result_dict["Calibrated retention time finish"] + RT_tol
        )
        RT_ref_act_peak = "Calibrated retention time"
    elif RT_ref == "pred":
        Maxquant_result_dict["RT_search_left"] = (
            Maxquant_result_dict["predicted_RT"] - RT_tol
        )
        Maxquant_result_dict["RT_search_right"] = (
            Maxquant_result_dict["predicted_RT"] + RT_tol
        )
        RT_ref_act_peak = "predicted_RT"
    elif RT_ref == "mix":
        Maxquant_result_dict["RT_search_left"] = (
            Maxquant_result_dict["Retention time new"] - RT_tol
        )
        Maxquant_result_dict["RT_search_right"] = (
            Maxquant_result_dict["Retention time new"] + RT_tol
        )
        RT_ref_act_peak = "Retention time new"
    try:  # try and read results
        scan_record = pd.read_pickle(output_file + "_scan_record.pkl")
        activation = np.load(output_file + "_activationByScanFromLasso.npy")

        logging.info("Load pre-calculated optimization.")
    except FileNotFoundError:
        logging.info("Precalculated optimization not found, start Scan By Scan.")
        logging.info("==================Scan By Scan==================")
        # Optimization
        start_time = time.time()
        logging.info("-----------------Scan by Scan Optimization-----------------")

        # process scans
        result_dict = process_scans_parallel(
            n_jobs=cpu_count(),
            MS1Scans=MS1Scans.iloc[
                1000:1010, :
            ],  # for small scale testing: MS1Scans.iloc[1000:1050, :]
            Maxquant_result=Maxquant_result_dict,
            loss="lasso",
            opt_algo=opt_algo,
            # alphas=[0],  # TODO: use alpha = 0 for threshold, change if needed!!
            alpha_criteria=alpha_criteria,
            AbundanceMissingThres=iso_ab_mis_thres,
            return_precursor_scan_cos_dist=PS_cos_dist,
        )
        minutes, seconds = divmod(time.time() - start_time, 60)
        logging.info(
            "Script execution time: {}m {}s".format(int(minutes), int(seconds))
        )
        # merge results
        n_ms1scans = MS1Scans.shape[0]
        activation = pd.DataFrame(
            index=Maxquant_result_dict["id"], columns=range(n_ms1scans)
        )
        precursor_scan_CosDist = pd.DataFrame(
            index=Maxquant_result_dict["id"], columns=range(n_ms1scans)
        )
        scan_record_list = []
        for scan_idx, result_dict_scan in result_dict.items():
            if result_dict_scan["activation"] is not None:
                activation.loc[
                    result_dict_scan["activation"]["precursor"], scan_idx
                ] = result_dict_scan["activation"]["activation"]
            if result_dict_scan["precursor_cos_dist"] is not None:
                precursor_scan_CosDist.loc[
                    result_dict_scan["precursor_cos_dist"]["precursor"], scan_idx
                ] = result_dict_scan["precursor_cos_dist"]["cos_dist"]
            scan_record_list.append(result_dict_scan["scans_record"])
        scan_record = pd.DataFrame(
            scan_record_list,
            columns=[
                "Scan",
                "Time",
                "CandidatePrecursorByRT",
                "FilteredPrecursor",
                "NumberHighlyCorrDictCandidate",
                "BestAlpha",
                "Cosine Dist",
                "IntensityExplained",
            ],
        )

        minutes, seconds = divmod(time.time() - start_time, 60)
        logging.info(
            "Script execution time: {}m {}s".format(int(minutes), int(seconds))
        )
        activation = activation.fillna(0)
        np.save(output_file + "_activationByScanFromLasso.npy", activation.values)
        if PS_cos_dist:
            precursor_scan_CosDist = precursor_scan_CosDist.fillna(0)
            np.save(
                output_file + "_precursor_scan_CosDist.npy",
                precursor_scan_CosDist.values,
            )
        scan_record.to_pickle(output_file + "_scan_record.pkl")
        minutes, seconds = divmod(time.time() - start_time, 60)
        logging.info(
            "Script execution time: {}m {}s".format(int(minutes), int(seconds))
        )

    logging.info("==================Post Processing==================")

    # calc activation sum w/o smoothing, w/ Gaussian smoothing and local minima smoothing
    MS1Scans_NoArray = pd.read_csv(os.path.join(dirname, MS1Scans_NoArray_name))
    try:
        sum_raw = pd.read_csv(os.path.join(result_dir, "sum_raw.csv"))
    except FileNotFoundError:
        _, sum_raw = post_processing.SmoothActivationMatrix(
            activation=activation, MS1Scans_noArray=MS1Scans_NoArray, method="Raw"
        )
        sum_raw.to_csv(os.path.join(result_dir, "sum_raw.csv"), index=False)

    try:
        refit_activation_minima = np.load(output_file + "_activationMinima.npy")
        sum_minima = pd.read_csv(os.path.join(result_dir, "sum_minima.csv"))
    except FileNotFoundError:
        refit_activation_minima, sum_minima = post_processing.SmoothActivationMatrix(
            activation=activation,
            MS1Scans_noArray=MS1Scans_NoArray,
            method="LocalMinima",
        )
        np.save(output_file + "_activationMinima.npy", refit_activation_minima)
        sum_minima.to_csv(os.path.join(result_dir, "sum_minima.csv"), index=False)

    try:
        refit_activation_gaussian = np.load(output_file + "_activationGaussian.npy")
        sum_gaussian = pd.read_csv(os.path.join(result_dir, "sum_gaussian.csv"))
    except FileNotFoundError:
        (
            refit_activation_gaussian,
            sum_gaussian,
        ) = post_processing.SmoothActivationMatrix(
            activation=activation,
            MS1Scans_noArray=MS1Scans_NoArray,
            method="GaussianKernel",
        )
        np.save(output_file + "_activationGaussian.npy", refit_activation_gaussian)
        sum_gaussian.to_csv(os.path.join(result_dir, "sum_gaussian.csv"), index=False)

    # Elution peak preservation
    try:
        sum_peak = pd.read_csv(os.path.join(result_dir, "sum_peak.csv"))
    except FileNotFoundError:
        sum_peak = post_processing.calculate_sum_activation_array(
            Maxquant_result=Maxquant_result_dict,
            MS1ScansNoArray=MS1Scans_NoArray,
            activation=refit_activation_minima,
            RT_ref=RT_ref_act_peak,
        )
        sum_peak.to_csv(os.path.join(result_dir, "sum_peak.csv"), index=False)

    logging.debug(
        "dimension of sum_raw, sum_gaussiam, sum_minima, sum_peak: %s, %s, %s, %s",
        sum_raw.shape,
        sum_gaussian.shape,
        sum_minima.shape,
        sum_peak.shape,
    )

    logging.info("==================Result Analaysis==================")
    SBS_result = result_analysis.SBSResult(
        ref_df=Maxquant_result_dict,
        exp_df=Maxquant_result_exp,
        sum_raw=sum_raw,
        sum_gaussian=sum_gaussian,
        sum_minima=sum_minima,
        sum_peak=sum_peak,
        RT_tol=RT_tol,
    )
    # Correlation
    for sum_col in SBS_result.SumActCol:
        SBS_result.plot_intensity_corr(
            inf_col=sum_col, interactive=False, save_dir=report_dir
        )

    # Overlap with MQ
    SBS_result.plot_overlap_with_MQ(save_dir=report_dir)

    # evaluate target and decoy
    SBS_result.eval_target_decoy(save_dir=report_dir)

    # Report
    scan_record = result_analysis.GenerateResultReport(
        scan_record=scan_record,
        intensity_cols=[SBS_result.ref_df[col] for col in SBS_result.SumActCol]
        + [SBS_result.ref_exp_df_inner["Intensity"]],
        save_dir=report_dir,
    )
    scan_record.to_csv(output_file + "_scan_record.csv")


if __name__ == "__main__":
    fire.Fire(opt_scan_by_scan)

"""Module providing a function calling the scan by scan optimization."""
import logging
import os

import time
from typing import Literal
import fire
import numpy as np
import pandas as pd
import sparse
import postprocessing.post_processing as post_processing
from postprocessing.peak_selection import match_peaks_to_exp
from utils.config import Config
from optimization.inference import process_ims_frames_parallel, generate_id_partitions
from result_analysis import result_analysis
import alphatims.bruker

# os.environ["NUMEXPR_MAX_THREADS"] = "8"


def _define_rt_search_range(
    maxquant_result_dict: pd.DataFrame,
    rt_tol: float,
    rt_ref: Literal["exp", "pred", "mix"],
):
    """Define the search range for the precursor RT."""
    if rt_ref == "exp":
        maxquant_result_dict["RT_search_left"] = (
            maxquant_result_dict["Calibrated retention time start"] - rt_tol
        )
        maxquant_result_dict["RT_search_right"] = (
            maxquant_result_dict["Calibrated retention time finish"] + rt_tol
        )
        rt_ref_act_peak = "Calibrated retention time"
    elif rt_ref == "pred":
        maxquant_result_dict["RT_search_left"] = (
            maxquant_result_dict["predicted_RT"] - rt_tol
        )
        maxquant_result_dict["RT_search_right"] = (
            maxquant_result_dict["predicted_RT"] + rt_tol
        )
        rt_ref_act_peak = "predicted_RT"
    elif rt_ref == "mix":
        maxquant_result_dict["RT_search_left"] = (
            maxquant_result_dict["Retention time new"] - rt_tol
        )
        maxquant_result_dict["RT_search_right"] = (
            maxquant_result_dict["Retention time new"] + rt_tol
        )
        rt_ref_act_peak = "Retention time new"
    maxquant_result_dict["RT_search_center"] = maxquant_result_dict[rt_ref_act_peak]
    return maxquant_result_dict


def _merge_activation_results(
    processed_scan_dict: dict, ref_id: pd.Series, n_ms1scans: int
):
    """Merge the activation results."""
    activation = pd.DataFrame(index=ref_id, columns=range(n_ms1scans))
    precursor_scan_cos_dist = pd.DataFrame(index=ref_id, columns=range(n_ms1scans))
    precursor_collinear_sets = pd.DataFrame(index=ref_id, columns=range(n_ms1scans))
    scan_record_list = []
    for scan_idx, result_dict_scan in processed_scan_dict.items():
        if result_dict_scan["activation"] is not None:
            activation.loc[
                result_dict_scan["activation"]["precursor"], scan_idx
            ] = result_dict_scan["activation"]["activation"]
        if result_dict_scan["precursor_cos_dist"] is not None:
            precursor_scan_cos_dist.loc[
                result_dict_scan["precursor_cos_dist"]["precursor"], scan_idx
            ] = result_dict_scan["precursor_cos_dist"]["cos_dist"]
        if result_dict_scan["precursor_collinear_sets"] is not None:
            precursor_collinear_sets.loc[
                result_dict_scan["precursor_collinear_sets"]["precursor"], scan_idx
            ] = result_dict_scan["precursor_collinear_sets"]["collinear_candidates"]
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
    return activation, precursor_scan_cos_dist, scan_record, precursor_collinear_sets


def opt_scan_by_scan(config_path: str):
    """Scan by scan optimization for joint identification and quantification."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    conf = Config(config_path)
    conf.make_result_dirs()

    # start analysis
    start_time_init = time.time()
    logging.info("==================Load data==================")

    # Load data
    maxquant_result_ref = pd.read_pickle(filepath_or_buffer=conf.mq_ref_path)
    maxquant_result_exp = pd.read_csv(conf.mq_exp_path, sep="\t", low_memory=False)
    data = alphatims.bruker.TimsTOF(conf.d_path)
    hdf_path = os.path.join(data.directory, f"{data.sample_name}.hdf")
    if not os.path.isfile(hdf_path):
        hdf_file_name = data.save_as_hdf(
            directory=data.directory, file_name=hdf_path, overwrite=False
        )
    else:
        logging.info("HDF file already exists")

    # ms1scans
    ms1scans = data.frames.loc[data.frames.MsMsType == 0]
    ms1scans["Time_minute"] = ms1scans["Time"] / 60
    ms1scans["MS1_frame_idx"] = (
        ms1scans["Time"].rank(axis=0, method="first", ascending=True).astype(int) - 1
    )
    ms1scans.set_index("MS1_frame_idx", inplace=True, drop=False)
    ms1scans.to_csv(os.path.join(conf.result_dir, "ms1scans.csv"))

    minutes, seconds = divmod(time.time() - start_time_init, 60)
    logging.info("Script execution time: %dm %ds", int(minutes), int(seconds))

    # mobilty values
    mobility_values = np.sort(data.mobility_values)
    mobility_values_df = pd.DataFrame(
        mobility_values, columns=["mobility_values"]
    ).reset_index()
    mobility_values_df.rename(columns={"index": "mobility_values_index"}, inplace=True)
    mobility_values_df.to_csv(os.path.join(conf.result_dir, "mobility_values.csv"))

    # add extra information into ref file, e.g. deifne RT search range
    maxquant_result_ref = _define_rt_search_range(
        maxquant_result_ref, conf.rt_tol, conf.rt_ref
    )
    maxquant_result_ref["mz_length"] = maxquant_result_ref["IsoMZ"].apply(
        lambda x: len(x)
    )
    maxquant_result_ref = maxquant_result_ref.sort_values("1/K0")
    maxquant_result_ref_with_im_index = pd.merge_asof(
        left=maxquant_result_ref,
        right=mobility_values_df,
        left_on="1/K0",
        right_on="mobility_values",
        direction="nearest",
    )
    maxquant_result_ref_with_im_index_sortmz = maxquant_result_ref_with_im_index.copy()
    maxquant_result_ref_with_im_index_sortmz["mz_rank"] = (
        maxquant_result_ref_with_im_index_sortmz["m/z"]
        .rank(axis=0, method="first", ascending=True)
        .astype(int)
    )
    maxquant_result_ref_with_im_index_sortmz.to_pickle(
        os.path.join(conf.result_dir, "maxquant_result_ref.pkl")
    )
    try:  # try and read results
        pept_act_sum_df = pd.read_csv(
            os.path.join(conf.result_dir, "pept_act_sum.csv"), index_col=0
        )
        logging.info("Loaded pre-calculated optimization.")
    except FileNotFoundError:
        logging.info("Precalculated optimization not found, start Scan By Scan.")
        logging.info("==================Scan By Scan==================")
        # Optimization
        start_time = time.time()
        logging.info("-----------------Scan by Scan Optimization-----------------")
        n_workers = conf.n_cpu
        n_batch = conf.n_batch
        logging.info("Number of batches: %s", n_batch)
        batch_scan_indices = generate_id_partitions(
            n_batch=n_batch,
            id_array=ms1scans.index.values[conf.start_frame : conf.end_frame],
            how="round_robin",
        )  # for small scale testing: ms1scans["Id"].iloc[0:500]
        logging.info("indices in first batch: %s", batch_scan_indices[0])
        # process scans
        process_ims_frames_parallel(
            data=data,
            n_jobs=n_workers,
            ms1scans=ms1scans,
            batch_scan_indices=batch_scan_indices,
            maxquant_ref=maxquant_result_ref_with_im_index_sortmz,
            mobility_values=mobility_values,
            delta_mobility_thres=conf.delta_mobility_thres,
            mz_bin_digits=conf.mz_bin_digits,
            process_in_blocks=True,
            width=conf.im_peak_extraction_width,
            path_prefix=conf.output_file,
            return_im_pept_act=True,
            extract_im_peak=False,
            n_blocks_by_pept= conf.n_blocks_by_pept
        )

        minutes, seconds = divmod(time.time() - start_time, 60)
        logging.info(
            "Process scans - Script execution time: %dm %ds", int(minutes), int(seconds)
        )

        # logging.info("=================Post Processing==================")

        for batch_num in range(conf.n_batch):
            # logging.info("Batch %s, output file path %s", batch_num, conf.output_file)
            act_3d = sparse.load_npz(
                conf.output_file + f"_im_rt_pept_act_coo_batch{batch_num}.npz"
            )
            pept_act_sum = act_3d.sum(axis=(0, 1))
            logging.info("NNZ size of batch %s act_3d %s", batch_num, act_3d.nnz)
            if batch_num == 0:
                act_3d_all = act_3d
                pept_act_sum_all = pept_act_sum
                del act_3d, pept_act_sum
            else:
                act_3d_all += act_3d
                pept_act_sum_all += pept_act_sum
                logging.info("NNZ size of act_3d_all %s", act_3d_all.nnz)
                del act_3d, pept_act_sum
        # pept_act_sum = act_3d_all.sum(axis=(0, 1))
        sparse.save_npz(
            conf.output_file + "pept_act_sum.npz",
            pept_act_sum_all,
        )
        pept_act_sum_array = sparse.asnumpy(pept_act_sum_all)
        del pept_act_sum_all
        pept_act_sum_df = pd.DataFrame(
            pept_act_sum_array[:],
            columns=["pept_act_sum"],
            index=np.arange(pept_act_sum_array.shape[0]),
        )
        pept_act_sum_df.to_csv(os.path.join(conf.result_dir, "pept_act_sum.csv"))

    logging.info("==================Result Analaysis==================")
    if "predicted_RT" not in maxquant_result_ref_with_im_index_sortmz.columns:
        maxquant_result_ref_with_im_index_sortmz[
            "predicted_RT"
        ] = maxquant_result_ref_with_im_index_sortmz["RT_search_center"]

    sbs_result = result_analysis.SBSResult(
        maxquant_ref_df=maxquant_result_ref_with_im_index_sortmz,
        maxquant_exp_df=maxquant_result_exp,
        sum_raw=pept_act_sum_df,
        ims=True,
    )

    sbs_result.compare_with_maxquant_exp_int(
        filter_by_rt_overlap=None, handle_mul_exp_pcm="drop", save_dir=conf.report_dir
    )
    merged_df = sbs_result.ref_exp_df_inner

    # peak_results_matched = match_peaks_to_exp(
    #     ref_exp_inner_df=merged_df, peak_results=peak_results
    # )
    # peak_results_matched.to_csv(
    #     os.path.join(conf.result_dir, "peak_results_matched.csv")
    # )

    # Correlation
    for sum_col in sbs_result.sum_cols:
        sbs_result.plot_intensity_corr(
            inf_col=sum_col, interactive=False, save_dir=conf.report_dir
        )

    # Overlap with MQ
    sbs_result.plot_overlap_with_MQ(save_dir=conf.report_dir, show_ref=True)

    # # evaluate target and decoy
    # sbs_result.eval_target_decoy(save_dir=conf.report_dir)

    # # selected alpha
    # if conf.opt_algo == "lasso_cd":
    #     result_analysis.plot_alphas_across_scan(
    #         scan_record=scan_record, x="Time", save_dir=conf.report_dir
    #     )

    # # Report
    # scan_record = result_analysis.generate_result_report(
    #     scan_record=scan_record,
    #     intensity_cols=[sbs_result.ref_df[col] for col in sbs_result.sum_cols]
    #     + [sbs_result.ref_exp_df_inner["Intensity"]],
    #     save_dir=conf.report_dir,
    # )
    # scan_record.to_csv(conf.output_file + "_scan_record.csv")


if __name__ == "__main__":
    fire.Fire(opt_scan_by_scan)

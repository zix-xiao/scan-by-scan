import os
import pandas as pd
from typing import Literal, Union
import logging
import pickle

Logger = logging.getLogger(__name__)
from deeplc import DeepLC
from deeplcretrainer import deeplcretrainer
import tensorflow as tf

from psm_utils.io.maxquant import MSMSReader
from psm_utils.io.peptide_record import to_dataframe
from utils.metrics import RT_metrics


from optimization.dictionary import calculate_modpept_isopattern


def generate_reference(
    for_train_filepath: str,
    to_pred_filepath: Union[str, None],  # used for prediction
    save_dir: Union[str, None] = None,
    train_frac: float = 0.9,
    update_model: Literal["calib", "transfer"] = "transfer",
    seed: Union[int, None] = None,
    train_suffix: Union[None, str] = None,
    pred_suffix: Union[None, str] = None,
    filter_by_RT_diff: Union[Literal["closest"], float, None] = "closest",
    save_model_name: Union[str, None] = None,
):
    """
    transfer learn DeepLC model and predict RT, and calculate isotope patterns

    :for_train_filepath: file containing data used for the training, calibration or transfer learning,
                         will be further split into train, val and test
    :to_pred_filepath: file containing data used for generating prediction, and further SBS analysis,
                        if None, use train data
    :save_dir: the directory to save all files generated, if not specified, use the parent dir of train
                and pred files, respectively
    :train_frac:
    :calib: whether to calibrate, if not, iRT results will be generated
    :transfer: whether to transfer learn
    :seed: set random seed

    """
    if seed is not None:
        tf.random.set_seed(seed)

    Logger.info("Num GPUs Available: %s ", len(tf.config.list_physical_devices("GPU")))

    # Load and prepare data
    (
        train_evidence_tranfer_file,
        train_evidence_file_transfer_pred,
        train_evidence_transfer_pred_Iso,
    ) = prepare_MQ_evidence(
        maxquant_evidence_filepath=for_train_filepath, suffix=train_suffix
    )
    train_MQ_peprec, train_peprec_agg = format_MQ_as_DeepLCinput(
        train_evidence_tranfer_file
    )

    if to_pred_filepath is None:
        pred_evidence_file_transfer_pred, pred_evidence_transfer_pred_Iso = (
            train_evidence_file_transfer_pred,
            train_evidence_transfer_pred_Iso,
        )
        pred_MQ_peprec, pred_peprec_agg = train_MQ_peprec, train_peprec_agg
    else:
        (
            pred_evidence_tranfer_file,
            pred_evidence_file_transfer_pred,
            pred_evidence_transfer_pred_Iso,
        ) = prepare_MQ_evidence(
            maxquant_evidence_filepath=to_pred_filepath, suffix=pred_suffix
        )
        Logger.info("pred_evidence_transfer_file is %s", pred_evidence_tranfer_file)
        pred_MQ_peprec, pred_peprec_agg = format_MQ_as_DeepLCinput(
            pred_evidence_tranfer_file
        )

    if save_dir is None:
        train_dir = os.path.dirname(for_train_filepath)
    else:
        train_dir = os.path.join(save_dir, "RT_tranfer_learn_train")
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
    # models
    prediction_dir = os.path.dirname(os.path.realpath(__file__))
    ori_model_paths = [
        "models/full_hc_train_pxd001468_1fd8363d9af9dcad3be7553c39396960.hdf5",
        "models/full_hc_train_pxd001468_8c22d89667368f2f02ad996469ba157e.hdf5",
        "models/full_hc_train_pxd001468_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
    ]
    ori_model_paths = [os.path.join(prediction_dir, dm) for dm in ori_model_paths]

    train_deeplc_file = os.path.join(train_dir, "train.csv")
    # train_evidence_file = os.path.join(train, "evidence_RT_pred.csv")

    # load DDA transfer learning data and split
    df_train = train_peprec_agg.sample(frac=train_frac, random_state=seed)
    df_test = train_peprec_agg.loc[train_peprec_agg.index.difference(df_train.index)]
    df_train.fillna("", inplace=True)
    df_test.fillna("", inplace=True)

    df_train.to_csv(
        train_deeplc_file, index=False
    )  # For training new models a file is needed

    match update_model:
        case "transfer":
            pred_col = "trans_pred_" + str(train_frac)

            # apply transfer learning
            models = deeplcretrainer.retrain(
                [train_deeplc_file],
                mods_transfer_learning=ori_model_paths,
                freeze_layers=True,
                n_epochs=10,
                freeze_after_concat=1,
            )

            # Make a DeepLC object with the models trained previously
            dlc = DeepLC(path_model=models, batch_num=1024000, pygam_calibration=False)
        case "calib":
            pred_col = "cal_pred_" + str(train_frac)

            # Call DeepLC with the downloaded models with GAM calibration
            dlc = DeepLC(
                path_model=ori_model_paths, batch_num=1024000, pygam_calibration=True
            )

    # Perform calibration, make predictions and calculate metrics
    dlc.calibrate_preds(seq_df=df_train)

    if save_model_name is not None:
        model_name = os.path.join(
            train_dir, save_model_name + "_" + update_model + ".pkl"
        )
        with open(model_name, "wb") as outp:
            pickle.dump(dlc, outp, pickle.HIGHEST_PROTOCOL)
    df_test[pred_col] = dlc.make_preds(seq_df=df_test)

    RTmetric = RT_metrics(RT_obs=df_test["tr"], RT_pred=df_test[pred_col])
    Logger.info("MAE: %s", RTmetric.CalcMAE())
    Logger.info("Delta RT 95 percent: %s", RTmetric.CalcDeltaRTwidth(95))
    Logger.info("Pearson Corr: %s", RTmetric.CalcPrsCorr())

    # prepare final prediction output
    pred_peprec_agg["predicted_RT"] = dlc.make_preds(seq_df=pred_peprec_agg)
    pred_filtered = match_pred_to_input(
        MQ_peprec=pred_MQ_peprec,
        peprec_RTpred=pred_peprec_agg,
        filter_by_RT_diff=filter_by_RT_diff,
    )
    pred_filtered.to_csv(pred_evidence_file_transfer_pred, sep="\t")

    pred_filtered["IsoMZ"], pred_filtered["IsoAbundance"] = zip(
        *pred_filtered.apply(
            lambda row: calculate_modpept_isopattern(
                modpept=row["Modified sequence"], charge=row["Charge"], ab_thres=0.01
            ),
            axis=1,
        )
    )
    pred_filtered.to_pickle(pred_evidence_transfer_pred_Iso)
    Logger.info(
        "Finish. Filtered prediction dataframe dimension: %s", pred_filtered.shape
    )

    return None


def prepare_MQ_evidence(
    maxquant_evidence_filepath: str, suffix: Union[None, str] = None
):
    maxquant_file_base = maxquant_evidence_filepath[:-4]
    if suffix is not None:
        maxquant_file_base += "_" + suffix
    maxquant_file_transfer = maxquant_file_base + "_RT_transfer.txt"
    maxquant_file_transfer_pred = maxquant_file_base + "_transfer_RT_pred_filtered.txt"
    maxquatn_file_transfer_pred_Iso = (
        maxquant_file_base + "_transfer_RT_pred_filtered_withIso.pkl"
    )
    evidence = pd.read_csv(maxquant_evidence_filepath, sep="\t")
    evidence = evidence.rename(columns={"MS/MS scan number": "Scan number"})
    # drop rows where Modified Sequence is NaN
    evidence = evidence.dropna(subset=["Modified sequence"])
    evidence.to_csv(maxquant_file_transfer, index=False, sep="\t")

    return (
        maxquant_file_transfer,
        maxquant_file_transfer_pred,
        maxquatn_file_transfer_pred_Iso,
    )


def format_MQ_as_DeepLCinput(maxquant_file_transfer: str):
    MQ = pd.read_csv(maxquant_file_transfer, sep="\t")
    reader = MSMSReader(maxquant_file_transfer)

    psm_list = reader.read_file()

    psm_list.add_fixed_modifications([("Carbamidomethyl", ["C"])])
    psm_list.apply_fixed_modifications()
    # Modify these to match the modifications in the data and library of deepLC model
    psm_list.rename_modifications({"ox": "Oxidation", "ac": "Acetyl", "Oxidation (M)": "Oxidation"})
    peprec = to_dataframe(psm_list)  # can be mapped to ori df

    # Only one RT for each (peptide seq, mod)
    peprec_agg = (
        peprec.groupby(by=["peptide", "modifications"])["observed_retention_time"]
        .median()
        .reset_index()
    )

    peprec_agg = peprec_agg.rename(
        columns={"peptide": "seq", "observed_retention_time": "tr"}
    )
    peprec_agg = peprec_agg[["seq", "modifications", "tr"]]
    MQ_peprec = pd.concat([MQ, peprec], axis=1)
    return MQ_peprec, peprec_agg


def match_pred_to_input(
    MQ_peprec: pd.DataFrame,
    peprec_RTpred: pd.DataFrame,
    filter_by_RT_diff: Union[Literal["closest"], float, None] = None,
):
    """

    :peprec_RTpred: the column containing RT prediction should be named 'predicted_RT'
    :filtered_by_RT_diff: whether and how to filter results based on RT difference
        'closest': only keeping precursors that elute the closest to predicted RT
        float: specify a threshold, all entries with difference larger will be discarded,
                and for the ones kept, intensity will be aggregated by sum
        None: do not filter
    """
    MQ_RTpred = pd.merge(
        left=MQ_peprec,
        right=peprec_RTpred,
        left_on=["peptide", "modifications"],
        right_on=["seq", "modifications"],
        how="left",
    )
    if filter_by_RT_diff is not None:
        MQ_RTpred["RT_diff"] = abs(
            MQ_RTpred["Retention time"] - MQ_RTpred["predicted_RT"]
        )

        if filter_by_RT_diff == "closest":
            n_before = MQ_RTpred.shape[0]
            MQ_RTpred = MQ_RTpred.loc[
                MQ_RTpred.groupby(["Modified sequence", "Charge"])["RT_diff"].idxmin()
            ]
            n_after = MQ_RTpred.shape[0]
            Logger.info("Removed %s entries.", n_before - n_after)

        elif isinstance(filter_by_RT_diff, float):
            Logger.debug("Filter by threshold %s", filter_by_RT_diff)
            n_before = MQ_RTpred.shape[0]
            MQ_RTpred = MQ_RTpred.loc[MQ_RTpred["RT_diff"] <= filter_by_RT_diff]
            n_after = MQ_RTpred.shape[0]
            Logger.info(
                "Removed %s entries from RT difference threshold", n_before - n_after
            )
            column_map = {col: "first" for col in MQ_RTpred.columns}
            column_map["Intensity"] = "sum"
            MQ_RTpred = MQ_RTpred.groupby(
                ["Modified sequence", "Charge"], as_index=False
            ).agg(column_map)
            n_after_after = MQ_RTpred.shape[0]
            Logger.info(
                "Removed %s entries from RT difference threshold",
                n_after - n_after_after,
            )

        else:
            raise ValueError(
                "filter_by_RT_diff should be either a float or str closest!"
            )

    return MQ_RTpred

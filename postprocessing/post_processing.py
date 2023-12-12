import logging
from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_widths
from sklearn.metrics import auc

Logger = logging.getLogger(__name__)

_SmoothMethods = Literal["GaussianKernel", "LocalMinima", "Raw"]


def calculate_sum_activation_array(
    Maxquant_result: pd.DataFrame,
    MS1ScansNoArray: pd.DataFrame,
    activation: np.ndarray,
    RT_ref: str = "Retention time new",
    n_peaks: int | None = 1,
    return_peak_result: bool = False,
    **kwargs,
):
    ref_RT_array = Maxquant_result[[RT_ref, "id"]].values.reshape(
        [activation.shape[0], 2]
    )
    Logger.debug("shape of activation %s", activation.shape)
    Logger.debug("shape of reference RT %s", ref_RT_array.shape)
    act_ref_RT = np.concatenate((activation, ref_RT_array), axis=1)
    Logger.debug("shape of act_ref_RT %s", act_ref_RT.shape)
    if return_peak_result:
        Logger.warning("Returning peak result is significantly slower!")
        results = [
            extract_elution_peak_from_row_act(
                activation_row=act_ref_RT_row[:-2],
                ref_RT=act_ref_RT_row[-2],
                PCM_id=act_ref_RT_row[-1],
                MS1ScansNoArray=MS1ScansNoArray,
                n_peaks=n_peaks,
                return_peak_result=True,
                **kwargs,
            )
            for act_ref_RT_row in act_ref_RT
        ]
        # logging.debug(results)
        peak_results, peak_sum_activation = zip(*results)
        sum_peak = pd.DataFrame({"AUCActivationPeak": peak_sum_activation})
        peak_results = pd.concat(peak_results, axis=0)
        return sum_peak, peak_results
    else:
        peak_sum_activation = np.apply_along_axis(
            lambda act_ref_RT_row: extract_elution_peak_from_row_act(
                activation_row=act_ref_RT_row[:-2],
                ref_RT=act_ref_RT_row[-2],
                PCM_id=act_ref_RT_row[-1],
                MS1ScansNoArray=MS1ScansNoArray,
                n_peaks=n_peaks,
                return_peak_result=False,
                **kwargs,
            ),
            axis=1,
            arr=act_ref_RT,
        )
        sum_peak = pd.DataFrame({"AUCActivationPeak": peak_sum_activation})
        return sum_peak


def extract_elution_peak_from_row_act(
    activation_row: Union[pd.Series, np.array],
    MS1ScansNoArray: pd.DataFrame,
    ref_RT: float,
    n_peaks: int = 1,
    return_peak_result: bool = False,
    peak_width_thres=(2, None),
    peak_height_thres=(None, None),
    PCM_id: int | None = 0,
    **kwargs,  # find_peaks parameters
):
    """ """
    # Logger.debug("row name %s", activation_row.name)
    peaks, peak_property = find_peaks(
        activation_row, width=peak_width_thres, height=peak_height_thres, **kwargs
    )
    left = np.round(peak_property["left_ips"], decimals=0).astype(int)
    right = np.round(peak_property["right_ips"], decimals=0).astype(int)
    Logger.debug("Reference retention time %s", ref_RT)
    peak_result = pd.DataFrame(
        {
            "id": np.repeat(PCM_id, len(left)).astype(int),
            "apex_scan": peaks,
            "apex_time": MS1ScansNoArray["starttime"][peaks].values,
            "start_scan": left,
            "start_time": MS1ScansNoArray["starttime"][left].values,
            "end_scan": right,
            "end_time": MS1ScansNoArray["starttime"][right].values,
            "peak_width": right - left,
            "peak_height": peak_property["peak_heights"],
            "peak_intensity_auc": [
                auc(
                    x=MS1ScansNoArray["starttime"][i - 1 : j + 1],
                    y=activation_row[i - 1 : j + 1],
                )
                for (i, j) in zip(left, right)
            ],
        }
    )
    if ref_RT is not None:
        # Logger.debug('Preserving the %s closest peaks to reference RT.', n_peaks)
        peak_result["RT_diff"] = abs(peak_result["apex_time"] - ref_RT)

        peak_result_preserved = peak_result.nsmallest(n_peaks, "RT_diff")
        sum_intensity = peak_result_preserved["peak_intensity_auc"].sum()
        Logger.debug("sum intensity %s", sum_intensity)
    else:
        sum_intensity = peak_result["peak_intensity_auc"].sum()

    if return_peak_result:
        return peak_result, sum_intensity
    else:
        return sum_intensity


# Define the Gaussian function
def Gauss(x, A, B):
    y = A * np.exp(-1 * B * x**2)
    return y


# Smooth results
def SmoothActivationCurve(
    PrecursorActivation,
    MS1Scan_noarray: pd.DataFrame,
    method: _SmoothMethods = "GaussianKernel",
    b=0.02,
    minima_width: int = 3,
    minima_prominence: int = 10000,
):
    replacedValues = []
    smoothedActivation = PrecursorActivation.copy()
    if PrecursorActivation.sum() > 0:
        non0idx = np.nonzero(PrecursorActivation)[0]
        if len(non0idx) > 2:
            match method:
                case "GaussianKernel":
                    gaussianKernelValue = np.zeros(shape=non0idx.shape)
                    for time in MS1Scan_noarray.loc[non0idx, "starttime"]:
                        gaussianKernelValue = np.exp(
                            -((MS1Scan_noarray.loc[non0idx, "starttime"] - time) ** 2)
                            / (2 * (b**2))
                        )
                        gaussianKernelValue /= gaussianKernelValue.sum()
                        replacedValues.append(
                            round(
                                PrecursorActivation[non0idx] * gaussianKernelValue
                            ).sum()
                        )
                    smoothedActivation[non0idx] = replacedValues
                    # smoothedActivation = np.array(smoothedActivation).reshape(-1)
                case "LocalMinima":
                    peaks, _ = find_peaks(
                        -PrecursorActivation, prominence=minima_prominence
                    )
                    if len(peaks) > 0:
                        (peakWidth, peakHeight, left, right) = peak_widths(
                            -PrecursorActivation, peaks, rel_height=1
                        )
                        # print(peakWidth, left, right)
                        for idx, width in enumerate(peakWidth):
                            # print(idx, width, peaks[idx], smoothedActivation[peaks[idx]] )
                            if width <= minima_width:
                                distance = [
                                    1 / abs(left[idx] - peaks[idx].item()),
                                    1 / abs(right[idx] - peaks[idx].item()),
                                ]
                                # print(distance)
                                smoothedActivation[peaks[idx]] = np.average(
                                    smoothedActivation[
                                        [
                                            int(np.floor(left[idx])),
                                            int(np.ceil(right[idx])),
                                        ]
                                    ],
                                    axis=0,
                                    weights=distance,
                                )
    return smoothedActivation


def SmoothActivationMatrix(
    activation: np.ndarray,  # TODO: doing more than 1 thing!
    MS1Scans_noArray: pd.DataFrame,
    method: _SmoothMethods = "GaussianKernel",
    b=0.02,
    minima_width: int = 3,
):
    # refit
    if method != "Raw":
        f = lambda row: SmoothActivationCurve(
            row,
            MS1Scan_noarray=MS1Scans_noArray,
            method=method,
            b=b,
            minima_width=minima_width,
        )
        refit_activation = np.apply_along_axis(func1d=f, axis=1, arr=activation)
    else:
        refit_activation = activation

    # Sum activation
    sumActivation = pd.DataFrame(
        {
            "SumActivation" + method: refit_activation.sum(axis=1),
            "AUCActivation"
            + method: np.apply_along_axis(
                lambda x: auc(MS1Scans_noArray["starttime"], x), 1, refit_activation
            ),
            "TRPZActivation"
            + method: np.apply_along_axis(
                lambda x: trapezoid(x=MS1Scans_noArray["starttime"], y=x),
                1,
                refit_activation,
            ),
        }
    )
    return refit_activation, sumActivation

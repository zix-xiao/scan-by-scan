""" config class for ScanByScan """
import json
import logging
import os
import pickle
from typing import Literal

Logger = logging.getLogger(__name__)

_alpha_opt_metric = Literal["cos_dist", "RMSE"]
_loss = Literal["lasso", "sqrt_lasso"]
_algo = ["lasso_lars", "lasso_cd", "lars", "omp", "threshold"]
_alpha_criteria = ["min", "convergence"]
_pp_method = Literal["raw", "sqrt"]
_AlignMethods = Literal["2stepNN", "peakRange"]


class Config:
    """Read config file and get information from it."""

    @property
    def mzml_path(self) -> str:
        """Get mzml path from config file."""
        return self.data.get("mzml_path")

    @property
    def mq_exp_path(self) -> str:
        """Get MQ experiment path from config file."""
        return self.data.get("MQ_exp_path")

    @property
    def mq_ref_path(self) -> str:
        """Get MQ reference path from config file."""
        return self.data.get("MQ_ref_path")

    @property
    def notes(self) -> str:
        """Get notes from config file."""
        return self.data.get("notes")

    @property
    def rt_tol(self) -> float:
        """Get RT tolerance from config file."""
        return self.data.get("RT_tol")

    @property
    def iso_ab_mis_thres(self) -> float:
        """Get isotope abundance missing threshold from config file."""
        return self.data.get("iso_ab_mis_thres", 0.5)

    @property
    def alpha_criteria(self) -> str:
        """Get alpha criteria from config file."""
        return self.data.get("alpha_criteria", "convergence")

    @property
    def rt_ref(self) -> str:
        """Get RT reference from config file."""
        return self.data.get("RT_ref")

    @property
    def peak_sel_cos_dist(self) -> bool:
        """Get peak selection cosine distance from config file."""
        return self.data.get("PS_cos_dist", False)

    @property
    def opt_algo(self) -> str:
        """Get optimization algorithm from config file."""
        return self.data.get("opt_algo", "threshold")

    @property
    def alphas(self) -> list:
        """Get alphas from config file."""
        if self.data["opt_algo"] == "threshold":
            self.data["alphas"] = []
            Logger.info("Threshold method selected, set alphas to empty list.")
        return self.data.get("alphas", None)

    @property
    def filename(self) -> str:
        """Get filename from config file."""
        self.data["filename"] = (
            self.notes
            + self.basename[:-5]
            + "_ScanByScan"
            + "_RTtol"
            + str(self.rt_tol)
            + "_"
            + self.opt_algo
            + "_missabthres"
            + str(self.iso_ab_mis_thres)
            + "_"
            + self.alpha_criteria
            + "_NoIntercept"
            + "_"
            + self.rt_ref
        )
        if self.peak_sel_cos_dist:
            self.data["filename"] += "_PScosDist"
        return self.data.get("filename")

    @property
    def result_dir(self) -> str:
        """Get result directory from config file."""
        self.data["result_dir"] = os.path.join(self.dirname, self.filename)
        return self.data.get("result_dir")

    @property
    def output_file(self) -> str:
        """Get output file from config file."""
        self.data["output_file"] = os.path.join(self.data["result_dir"], "output")
        return self.data.get("output_file")

    @property
    def dirname(self) -> str:
        """Get dirname from config file."""
        self.data["dirname"] = os.path.dirname(self.data["mzml_path"])
        return self.data.get("dirname")

    @property
    def basename(self) -> str:
        """Get basename from config file."""
        self.data["basename"] = os.path.basename(self.data["mzml_path"])
        return self.data.get("basename")

    @property
    def ms1scans_no_array_name(self) -> str:
        """Get ms1scans_no_array_name from config file."""
        self.data["ms1scans_no_array_name"] = (
            self.basename[:-5] + "_MS1Scans_NoArray.csv"
        )
        return self.data.get("ms1scans_no_array_name")

    @property
    def report_dir(self) -> str:
        """Get report directory from config file."""
        self.data["report_dir"] = os.path.join(self.data["result_dir"], "report")
        return self.data.get("report_dir")

    @property
    def rt_ref_act_peak(self) -> str:
        """Get RT reference elution peak from RT reference method"""
        match self.rt_ref:
            case "mix":
                self.data["RT_ref_act_peak"] = "Retention time new"
            case "pred":
                self.data["RT_ref_act_peak"] = "predicted_RT"
            case "exp":
                self.data["RT_ref_act_peak"] = "Calibrated retention time"
        return self.data.get("RT_ref_act_peak")

    ########################
    # functions start here #
    ########################

    def __init__(self, config_path):
        self.path = config_path
        self.data = self.read_config()
        self.check_config()

    def read_config(self):
        """Read config file."""
        with open(self.path, mode="r", encoding="utf-8") as file:
            config_data = json.load(file)

        return config_data

    def check_config(self):
        """Check config file validation."""
        if self.data["opt_algo"] not in _algo:
            raise AssertionError(
                f"Invalid optimization algorithm, should be one of: {_algo}"
            )
        if self.data["alpha_criteria"] not in _alpha_criteria:
            raise AssertionError(
                f"Invalid alpha criteria, should be one of: {_alpha_criteria}"
            )

    def make_result_dirs(self):
        """Make result directory from config."""
        try:
            Logger.info(
                "Use RT reference method %s and RT elution peak %s",
                self.rt_ref,
                self.rt_ref_act_peak,
            )
            Logger.info("Result directory: %s", self.result_dir)
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)

            if not os.path.exists(self.report_dir):
                os.makedirs(self.report_dir)
                os.makedirs(os.path.join(self.report_dir, "activation"))

            json_saved_args = json.dumps(self.data)
            with open(
                os.path.join(self.result_dir, "param.json"),
                mode="w",
                encoding="utf-8",
            ) as f:
                f.write(json_saved_args)

            with open(os.path.join(self.result_dir, "param.pkl"), "wb") as f:
                pickle.dump(self.data, f)
        except AttributeError as e:
            raise AttributeError("Please set mzml_path first.") from e

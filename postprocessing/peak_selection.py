import logging
import pathlib
from typing import Literal, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    Masking,
)
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from numba import jit

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (
    StandardScaler,
    minmax_scale,
    scale,
)
from keras import metrics, regularizers
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.models import load_model
from utils.tools import _perc_fmt
from utils.plot import plot_pie, save_plot
import seaborn as sns

Logger = logging.getLogger(__name__)

METRICS = [
    metrics.BinaryCrossentropy(name="cross entropy"),  # same as model's loss
    metrics.MeanSquaredError(name="Brier score"),
    metrics.TruePositives(name="tp"),
    metrics.FalsePositives(name="fp"),
    metrics.TrueNegatives(name="tn"),
    metrics.FalseNegatives(name="fn"),
    metrics.BinaryAccuracy(name="accuracy"),
    metrics.Precision(name="precision"),
    metrics.Recall(name="recall"),
    metrics.AUC(name="auc"),
    metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
]


def match_peaks_to_exp(
    ref_exp_inner_df: pd.DataFrame,
    peak_results: pd.DataFrame,
    n_peaks: int = 1,
):
    """Match peaks to experimental peaks and return True or False in "matched" column"""
    # TODO: currently only supports one exp pcm RT per precursor id
    peak_results = pd.merge(
        left=peak_results,
        right=ref_exp_inner_df[
            [
                "id",
                "Calibrated retention time start",
                "Calibrated retention time",
                "Calibrated retention time finish",
            ]
        ],
        on="id",
        how="left",
    )
    peak_results["RT_apex_diff"] = abs(
        peak_results["apex_time"] - peak_results["Calibrated retention time"]
    )

    Logger.info("Use RT range as reference for peak selection.")
    peak_results["RT_start_diff"] = abs(
        peak_results["start_time"] - peak_results["Calibrated retention time start"]
    )
    peak_results["RT_end_diff"] = abs(
        peak_results["end_time"] - peak_results["Calibrated retention time finish"]
    )
    peak_results["RT_diff_sum"] = (
        peak_results["RT_start_diff"] + peak_results["RT_end_diff"]
    )

    # drop rows where "RT_diff_sum" is NaN
    peak_results.dropna(subset=["RT_diff_sum"], inplace=True)

    peak_results.loc[
        peak_results.groupby("id")["RT_diff_sum"]
        .nsmallest(n_peaks)
        .index.get_level_values(1),
        "matched",
    ] = True
    peak_results["matched"].fillna(False, inplace=True)

    return peak_results


def peak_results_report(peak_results: pd.DataFrame, save_dir: str | None = None):
    """Generate a report for the peak results dataframe"""
    # Calculate width distribution for matched and unmatched peaks
    matched_widths = peak_results[peak_results["matched"]]["peak_width"]
    unmatched_widths = peak_results[~peak_results["matched"]]["peak_width"]

    plot_pie(
        sizes=matched_widths.value_counts().values,
        labels=matched_widths.value_counts().index.values,
        title="Peak Width Distribution for Matched Peaks",
        save_dir=save_dir,
        fig_spec_name="MatchedPeakWidth",
        accumulative_threshold=0.95,
    )

    # Generate pie chart for unmatched peaks
    plot_pie(
        sizes=unmatched_widths.value_counts().values,
        labels=unmatched_widths.value_counts().index.values,
        title="Peak Width Distribution for Unmatched Peaks",
        save_dir=save_dir,
        fig_spec_name="UnmatchedPeakWidth",
        accumulative_threshold=0.95,
    )

    # Get the top 10 width value counts
    top_widths = peak_results["peak_width"].value_counts().nlargest(10)

    # Plot the count plot for the top 10 widths
    sns.countplot(
        data=peak_results[peak_results["peak_width"].isin(top_widths.index)],
        x="peak_width",
        hue="matched",
    )

    # Set the title and labels
    plt.title("Peak Width Distribution")
    plt.xlabel("Peak Width")
    plt.ylabel("Count")

    save_plot(save_dir, fig_type_name="Count", fig_spec_name="PeakWidth")


def transform_activation(
    activation_df: pd.DataFrame,
    log_intensity: bool = False,
    standardize: Literal["std", "minmax"] | None = "std",
):
    """Transform the activation dataframe to a standardized (and/or log) dataframe"""
    activation_values = activation_df.values
    if log_intensity:
        activation_values = np.log(activation_values + 1)
    match standardize:
        case "std":
            activation_values_std = scale(activation_values, axis=1)
        case "minmax":
            activation_values_std = minmax_scale(
                activation_values, axis=1, feature_range=(0, 1)
            )
        case None:
            activation_values_std = activation_values
    activation_df_std = pd.DataFrame(
        data=np.array(activation_values_std),
        index=activation_df.index,
        columns=activation_df.columns,
    )
    return activation_df_std


def prepare_peak_input(
    peak_results: pd.DataFrame,
    activation_df: pd.DataFrame,
    log_intensity: bool = False,
    standardize: Literal["std", "minmax"] | None = "std",
    margin: int = 3,
    method: Literal["mask", "clip"] = "mask",
):
    """Prepare the peak input as arrays for peak classification"""
    activation_df_std = transform_activation(activation_df, log_intensity, standardize)

    def _get_roi(peak_results_row: pd.Series, method: Literal["mask", "clip"] = "mask"):
        """Get the region of interest from the activation dataframe"""
        start_index = peak_results_row["start_scan"] - margin
        end_index = peak_results_row["end_scan"] + margin
        match method:
            case "mask":  # get roi by masking out the rest
                original_row = activation_df_std.loc[peak_results_row["id"]].values
                new_row = np.zeros_like(original_row)
                new_row[start_index : end_index + 1] = original_row[
                    start_index : end_index + 1
                ]
                return new_row[
                    peak_results_row["sbs_window_left_scan"] : peak_results_row[
                        "sbs_window_right_scan"
                    ]
                    + 1
                ]
            case "clip":  # get roi by clipping the rest
                return activation_df_std.loc[
                    peak_results_row["id"],
                    start_index : end_index + 1,
                ].values

    peaks = peak_results.apply(_get_roi, axis=1, method=method)
    return peaks.values


def prepare_seq_input_label(
    activation_df: pd.DataFrame,
    ref_dict_exp_inner_df: pd.DataFrame,
    log_intensity: bool = False,
    standardize: Literal["std", "minmax"] | None = "std",
):
    activation_df_std = transform_activation(activation_df, log_intensity, standardize)
    ref_dict_exp_inner_df.index = ref_dict_exp_inner_df.index.astype(int)

    def mark_peak(ref_dict_exp_inner_df_row):
        precursor_id = ref_dict_exp_inner_df_row.id
        (
            sbs_window_left_scan,
            sbs_window_right_scan,
        ) = ref_dict_exp_inner_df_row.loc[
            ["sbs_window_left_scan", "sbs_window_right_scan"]
        ].astype(int)
        label = np.zeros_like(activation_df_std.iloc[0].values)
        label[
            ref_dict_exp_inner_df_row[
                "Calibrated retention time start_scan"
            ] : ref_dict_exp_inner_df_row["Calibrated retention time finish_scan"]
        ] = 1
        activation_df_row = activation_df_std.loc[precursor_id]
        return (
            label[sbs_window_left_scan : sbs_window_right_scan + 1],
            activation_df_row[sbs_window_left_scan : sbs_window_right_scan + 1].values,
        )

    all_results = ref_dict_exp_inner_df.apply(mark_peak, axis=1, result_type="expand")
    label = all_results[0].values
    seq = all_results[1].values
    return label, seq


class PeakSegModel:
    def __init__(
        self,
        seq_input: np.ndarray,
        seq_label: np.ndarray,
    ):
        self.seq_input = seq_input
        self.seq_label = seq_label

    def pad_peak_input(
        self,
        maxlen: int | None = None,
        value: float = -1.0,
        padding: Literal["post", "two_side"] = "post",
    ):
        if maxlen is None:
            maxlen = max(map(len, self.seq_input))
        self.maxlen = maxlen
        Logger.debug("Max length: %s", maxlen)
        match padding:
            case "post":

                @jit(nopython=True)
                def pad_sequence(peak, maxlen, value):
                    result = np.full((maxlen,), value)
                    result[: len(peak)] = peak
                    return result

            case "two_side":

                @jit(nopython=True)
                def pad_sequence(peak, maxlen, value):
                    result = np.full((maxlen,), value)
                    pad_length = maxlen - len(peak)
                    before = pad_length // 2
                    result[before : before + len(peak)] = peak
                    return result

        self.padded_sequences_input = np.array(
            [pad_sequence(peak, maxlen, value) for peak in self.seq_input]
        ).astype(np.float32)

        self.padded_sequences_label = np.array(
            [pad_sequence(peak, maxlen, 0) for peak in self.seq_label]
        ).astype(np.float32)

        # Logger.debug("Padded sequences type: %s", self.padded_sequences_input.dtype)

    def plot_sbswindow_with_mask(self, row_id: int):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax2 = ax.twinx()
        ax.plot(self.padded_sequences_input[row_id], color="blue")
        ax2.plot(self.padded_sequences_label[row_id], color="red")
        plt.show()


class PeakClsModel:
    def __init__(
        self,
        peak_input: np.ndarray,
        peak_results: pd.DataFrame | None = None,
        peak_label: np.ndarray | None = None,
        precursor_id: np.ndarray | None = None,
        random_seed: int = 42,
        initial_bias: Union[float, None, Literal["auto"]] = None,
        use_class_weight: bool = False,
    ):
        self.peak_input = peak_input
        self.random_seed = random_seed

        if peak_results is not None:
            self.peak_results = peak_results
            self.peak_label = peak_results["matched"]
            self.precursor_id = peak_results["id"]
        else:
            assert peak_label is not None
            assert precursor_id is not None
            self.peak_label = peak_label
            self.precursor_id = precursor_id
        self.label_list = sorted(list(set(self.peak_label)))
        self.padded_sequences = None
        self.mask = None
        self.initial_bias = initial_bias
        self.use_class_weight = use_class_weight
        self.maxlen = None

    def report_data_distribution(self):
        print("Number of precursors:", len(set(self.precursor_id)))
        print(
            "Label"
            f" distribution:\n{self.peak_results['matched'].value_counts().to_string()}"
        )
        print(f"All peak results:\n{self.peak_results.describe().to_string()}")
        print("Positive peak results:")
        print(
            self.peak_results[self.peak_results["matched"] == 1].describe().to_string()
        )
        print("Negative peak results:")
        print(
            self.peak_results[self.peak_results["matched"] == 0].describe().to_string()
        )
        print("Number of peaks for each precursor:")
        print(self.peak_results.groupby("id").size().describe().to_string())
        self.peak_results.groupby("id").size().hist(
            bins=self.peak_results.groupby("id").size().max()
        )

    def pad_peak_input(
        self,
        maxlen: int | None = None,
        value: float = -1.0,
        padding: Literal["post", "two_side"] = "post",
    ):
        if maxlen is None:
            maxlen = max(map(len, self.peak_input))
        self.maxlen = maxlen
        match padding:
            case "post":

                @jit(nopython=True)
                def pad_sequence(peak, maxlen, value):
                    result = np.full((maxlen,), value)
                    result[: len(peak)] = peak
                    return result

            case "two_side":

                @jit(nopython=True)
                def pad_sequence(peak, maxlen, value):
                    result = np.full((maxlen,), value)
                    pad_length = maxlen - len(peak)
                    if pad_length >= 0:
                        before = pad_length // 2
                        # after = pad_length - before
                        result[before : before + len(peak)] = peak
                    if pad_length < 0:
                        print(
                            "peak input longer than maxlen, peak input will be clipped!"
                        )
                        before = abs(pad_length) // 2
                        # after = pad_length - before
                        result = peak[before : before + pad_length]
                    return result

        self.padded_sequences = np.array(
            [pad_sequence(peak, maxlen, value) for peak in self.peak_input]
        ).astype(np.float32)
        Logger.debug("Padded sequences type: %s", self.padded_sequences.dtype)

    def split_data(self, by_precrusor: bool = True, normalize: bool = True):
        np.random.seed(self.random_seed)
        if by_precrusor:
            unique_ids = list(set(self.precursor_id))
            np.random.shuffle(unique_ids)

            self.train_ids = unique_ids[: int(len(unique_ids) * 0.8)]
            self.test_ids = unique_ids[int(len(unique_ids) * 0.8) :]

            self.train_indices = np.where(self.precursor_id.isin(self.train_ids))[0]
            Logger.debug(
                "Train indices top 5 max: %s",
                (-self.train_indices).argsort()[:5],
            )
            self.test_indices = np.where(self.precursor_id.isin(self.test_ids))[0]
        else:
            indices = np.arange(len(self.padded_sequences))
            np.random.shuffle(indices)

            self.train_indices = indices[: int(len(indices) * 0.8)]
            self.test_indices = indices[int(len(indices) * 0.8) :]

        X_train = self.padded_sequences[self.train_indices]
        X_test = self.padded_sequences[self.test_indices]
        Logger.debug("X_train shape: %s", X_train.shape)
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        self.X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        self.X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        Logger.debug("X_train type: %s", self.X_train.dtype)
        Logger.debug("X_train shape: %s", self.X_train.shape)

        self.y_train = self.peak_label[self.train_indices].values
        Logger.debug("Y_train type: %s", self.X_train.dtype)
        Logger.debug("Y_train shape: %s", self.y_train.shape)

        self.y_test = self.peak_label[self.test_indices].values

        neg, pos = np.bincount(self.y_train)
        self.neg, self.pos = neg, pos
        total = neg + pos
        if self.initial_bias == "auto":
            print(
                "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of"
                " total)\n".format(total, pos, 100 * pos / total)
            )
            self.initial_bias = np.log([pos / neg])
            Logger.debug("Automatically calculate output bias: %s", self.initial_bias)
        if self.use_class_weight:
            # Scaling by total/2 helps keep the loss to a similar magnitude.
            # The sum of the weights of all examples stays the same.
            weight_for_0 = (1 / neg) * (total / 2.0)
            weight_for_1 = (1 / pos) * (total / 2.0)

            self.class_weight = {0: weight_for_0, 1: weight_for_1}
        else:
            self.class_weight = None
        self.resampled_ds = None

    def resample_train_data(
        self,
        resample_method: Literal["oversample", "undersample"],
        BUFFER_SIZE: int = 100000,
        BATCH_SIZE: int = 256,
    ):
        self.BATCH_SIZE = BATCH_SIZE
        pos_features = self.X_train[self.y_train == 1]
        neg_features = self.X_train[self.y_train == 0]
        pos_labels = self.y_train[self.y_train == 1]
        neg_labels = self.y_train[self.y_train == 0]
        match resample_method:
            case "oversample":  # oversample the minority class (positive class)
                ids = np.arange(len(pos_features))
                choices = np.random.choice(ids, len(neg_features))

                res_pos_features = pos_features[choices]
                res_pos_labels = pos_labels[choices]

                Logger.debug(
                    "Oversampling positive features, resampled positive"
                    " features shape: %s",
                    res_pos_features.shape,
                )

                resampled_features = np.concatenate(
                    [res_pos_features, neg_features], axis=0
                )
                resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)
            case "undersample":  # undersample the majority class (negative class)
                ids = np.arange(len(neg_features))
                choices = np.random.choice(ids, len(pos_features))

                res_neg_features = neg_features[choices]
                res_neg_labels = neg_labels[choices]

                Logger.debug(
                    "Undersampling negative features, resampled negative"
                    " features shape: %s",
                    res_neg_features.shape,
                )

                resampled_features = np.concatenate(
                    [res_neg_features, pos_features], axis=0
                )
                resampled_labels = np.concatenate([res_neg_labels, pos_labels], axis=0)

        order = np.arange(len(resampled_labels))
        np.random.shuffle(order)
        resampled_features = resampled_features[order]
        resampled_labels = resampled_labels[order]

        Logger.debug("Resampled features shape: %s", resampled_features.shape)

        def make_ds(features, labels):
            ds = tf.data.Dataset.from_tensor_slices((features, labels))  # .cache()
            ds = ds.shuffle(BUFFER_SIZE).repeat()
            return ds

        pos_ds = make_ds(pos_features, pos_labels)
        neg_ds = make_ds(neg_features, neg_labels)

        for features, label in pos_ds.take(1):
            Logger.debug("Features shape: %s", features.shape)
            Logger.debug("Label: %s", label.numpy())

        resampled_ds = tf.data.Dataset.sample_from_datasets(
            [pos_ds, neg_ds], weights=[0.5, 0.5]
        )
        self.resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)

        for features, label in self.resampled_ds.take(1):
            Logger.debug("Label mean: %s", label.numpy().mean())
        self.resampled_steps_per_epoch = np.ceil(2.0 * self.neg / self.BATCH_SIZE)
        val_ds = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test)).cache()
        self.val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

    def make_model(
        self,
        config: dict,
    ):
        output_bias = None

        if isinstance(self.initial_bias[0], float) and self.resampled_ds is None:
            output_bias = Constant(self.initial_bias)
            Logger.debug("Output bias: %s", output_bias)

        # if model then load existing trained model
        match config["model"]:
            case "LSTM":
                self.y_train = self.y_train.reshape(-1, 1)
                # Build the LSTM model with Masking layer
                model = Sequential()
                model.add(
                    Masking(mask_value=-1)
                )  # Masking layer with the padding marker
                model.add(
                    LSTM(
                        config["lstm_n_units"],
                        activation="relu",
                        input_shape=(1, self.maxlen),
                    )
                )
                model.add(
                    Dense(
                        1,
                        activation="sigmoid",
                        use_bias=True,
                        bias_initializer=output_bias,
                        name="dense_prediction",
                    )
                )

                model.compile(
                    optimizer=Adam(learning_rate=config["learning_rate"]),
                    loss=BinaryCrossentropy(),
                    metrics=config["metric"],
                )
                self.class_model = model
            case "CNN":
                # Build the CNN model with Masking layer
                model = Sequential()
                model.add(
                    Masking(mask_value=-1)
                )  # Masking layer with the padding marker
                model.add(
                    Conv1D(
                        filters=config["conv1_n_filters"],
                        kernel_size=config["conv1_kernel_size"],
                        activation="relu",
                        input_shape=(None, self.maxlen),
                    )
                )
                model.add(
                    Conv1D(
                        filters=config["conv2_n_filters"],
                        kernel_size=config["conv2_kernel_size"],
                        activation="relu",
                        input_shape=(None, self.maxlen),
                    )
                )
                model.add(GlobalMaxPooling1D())
                model.add(
                    Dense(
                        config["dense1_n"],
                        kernel_regularizer=regularizers.l2(config["dense1_reg_rate"]),
                        activation="relu",
                        name="dense_1",
                    )
                )
                model.add(Dropout(config["dropout_rate"], name="dropout_1"))
                model.add(
                    Dense(
                        1,
                        activation="sigmoid",
                        use_bias=True,
                        bias_initializer=output_bias,
                        name="dense_prediction",
                    )
                )

                model.compile(
                    optimizer=Adam(learning_rate=config["learning_rate"]),
                    loss=BinaryCrossentropy(),
                    metrics=METRICS,
                )
                self.class_model = model
            case _:
                model_path = pathlib.Path(model)
                Logger.info("Using existing model: %s", model_path.resolve())
                self.class_model = load_model(model)

    def train_model(self, epochs=1000, weight_class: bool = False, **kwargs):
        # Define early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_prc",
            verbose=1,
            patience=10,
            mode="max",
            restore_best_weights=True,
        )

        Logger.info("Class weights: %s", self.class_weight)
        if self.resampled_ds is not None:
            Logger.info("Use balanced dataset for training, set bias to zero")
            self.class_model.fit(
                self.resampled_ds,
                validation_data=self.val_ds,
                epochs=epochs,
                steps_per_epoch=self.resampled_steps_per_epoch,
                callbacks=[early_stopping],
                class_weight=self.class_weight,
                batch_size=self.BATCH_SIZE,
                **kwargs,
            )
        else:
            self.class_model.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_test, self.y_test),
                epochs=epochs,
                callbacks=[early_stopping],
                class_weight=self.class_weight,
                **kwargs,
            )

    def evaluate_id_based_cls(self, top_n: int = 1):
        # Predict probabilities
        y_test_pred_prob = self.class_model.predict(self.X_test)

        # Create a DataFrame with precursor_ids, predicted probabilities, and actual labels
        df_test = pd.DataFrame(
            {
                "precursor_id": self.precursor_id[self.test_indices],
                "y_pred_prob": y_test_pred_prob.flatten(),
                "y_true": self.y_test,
            }
        )
        if top_n == 1:
            # For each precursor_id, assign 1 to the highest probability and 0 to the rest
            df_test["y_pred"] = (
                df_test.groupby("precursor_id")["y_pred_prob"]
                .transform(lambda x: x == x.max())
                .astype(int)
            )
        else:
            # For each precursor_id, assign 1 to the top_n highest probabilities and 0 to the rest
            df_test["rank"] = df_test.groupby("precursor_id")["y_pred_prob"].rank(
                method="first", ascending=False
            )
            df_test["y_pred"] = (df_test["rank"] <= top_n).astype(int)
            df_test.drop(columns=["rank"], inplace=True)

        # only keep the true peak of each precursor_id
        self.df_test = df_test.copy()
        df_test = df_test[df_test["y_true"] == 1]

        # Calculate accuracy
        accuracy = accuracy_score(df_test["y_true"], df_test["y_pred"])

        return accuracy

    def get_model_summary(self):
        self.class_model.summary()

    def save_model(self, model_path: pathlib.Path):
        self.class_model.save(model_path.resolve())
        Logger.info("Saved model to: %s", model_path.resolve())


def model_tuner(hp, maxlen: int = 361, initial_bias: List[float] | None = -1.04778782):
    model = Sequential()
    model.add(Masking(mask_value=-1))  # Masking layer with the padding marker
    hp_conv1_n_filters = hp.Int("conv1_n_filters", min_value=8, max_value=128, step=8)
    hp_conv1_kernel_size = hp.Int("conv1_kernel_size", min_value=3, max_value=9, step=2)
    model.add(
        Conv1D(
            filters=hp_conv1_n_filters,
            kernel_size=hp_conv1_kernel_size,
            activation="relu",
            input_shape=(None, maxlen),
        )
    )
    hp_conv2_n_filters = hp.Int("conv2_n_filters", min_value=8, max_value=128, step=8)
    hp_conv2_kernel_size = hp.Int("conv2_kernel_size", min_value=3, max_value=9, step=2)
    model.add(
        Conv1D(
            filters=hp_conv2_n_filters,
            kernel_size=hp_conv2_kernel_size,
            activation="relu",
            input_shape=(None, maxlen),
        )
    )
    model.add(GlobalMaxPooling1D())
    hp_dense1_n = hp.Int("dense1_n", min_value=8, max_value=128, step=8)
    hp_dense1_reg_rate = hp.Choice("dense1_reg_rate", [0.0, 0.1, 0.01, 0.001, 0.0001])
    model.add(
        Dense(
            hp_dense1_n,
            kernel_regularizer=regularizers.l2(hp_dense1_reg_rate),
            activation="relu",
            name="dense_1",
        )
    )
    hp_dropout_rate = hp.Choice("dropout_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    model.add(Dropout(hp_dropout_rate, name="dropout_1"))
    if isinstance(initial_bias, float):
        output_bias = Constant(initial_bias)
        Logger.debug("Output bias: %s", output_bias)
    else:
        output_bias = None
    model.add(
        Dense(
            1,
            activation="sigmoid",
            bias_initializer=output_bias,
            name="dense_prediction",
        )
    )

    hp_learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss=BinaryCrossentropy(),
        metrics=METRICS,
    )
    return model


def model_tune_wandb(config, maxlen, output_bias: float | None = None):
    model = Sequential()
    model.add(Masking(mask_value=-1))  # Masking layer with the padding marker
    model.add(
        Conv1D(
            filters=config["conv1_n_filters"],
            kernel_size=config["conv1_kernel_size"],
            activation="relu",
            input_shape=(None, maxlen),
        )
    )
    model.add(
        Conv1D(
            filters=config["conv2_n_filters"],
            kernel_size=config["conv2_kernel_size"],
            activation="relu",
            input_shape=(None, maxlen),
        )
    )
    model.add(GlobalMaxPooling1D())
    model.add(
        Dense(
            config["dense1_n"],
            kernel_regularizer=regularizers.l2(config["dense1_reg_rate"]),
            activation="relu",
            name="dense_1",
        )
    )
    model.add(Dropout(config["dropout_rate"], name="dropout_1"))
    model.add(
        Dense(
            1,
            activation="sigmoid",
            bias_initializer=output_bias,
            name="dense_prediction",
        )
    )
    return model


def evaluate_id_based_cls(class_model, mdl, top_n: int = 1):
    # Predict probabilities
    y_test_pred_prob = class_model.predict(mdl.X_test)

    # Create a DataFrame with precursor_ids, predicted probabilities, and actual labels
    df_test = pd.DataFrame(
        {
            "precursor_id": mdl.precursor_id[mdl.test_indices],
            "y_pred_prob": y_test_pred_prob.flatten(),
            "y_true": mdl.y_test,
        }
    )
    if top_n == 1:
        # For each precursor_id, assign 1 to the highest probability and 0 to the rest
        df_test["y_pred"] = (
            df_test.groupby("precursor_id")["y_pred_prob"]
            .transform(lambda x: x == x.max())
            .astype(int)
        )
    else:
        # For each precursor_id, assign 1 to the top_n highest probabilities and 0 to the rest
        df_test["rank"] = df_test.groupby("precursor_id")["y_pred_prob"].rank(
            method="first", ascending=False
        )
        df_test["y_pred"] = (df_test["rank"] <= top_n).astype(int)
        df_test.drop(columns=["rank"], inplace=True)

    # only keep the true peak of each precursor_id
    mdl.df_test = df_test.copy()
    df_test = df_test[df_test["y_true"] == 1]

    # Calculate accuracy
    accuracy = accuracy_score(df_test["y_true"], df_test["y_pred"])

    return accuracy, df_test

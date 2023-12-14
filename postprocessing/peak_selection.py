import logging
import pathlib
import random
from typing import Literal, Union

import numpy as np
import pandas as pd
from keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Masking,
    LSTM,
)
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from numba import jit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    scale,
    minmax_scale,
)
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from tensorflow.keras.initializers import Constant
import tensorflow as tf
import matplotlib.pyplot as plt

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


def prepare_peak_input(
    peak_results: pd.DataFrame,
    activation_df: pd.DataFrame,
    log_intensity: bool = False,
    standardize: Literal["std", "minmax"] | None = "std",
    margin: int = 3,
    method: Literal["mask", "clip"] = "mask",
):
    if log_intensity:
        activation_df = np.log(activation_df + 1)
    match standardize:
        case "std":
            activation_df_std = scale(activation_df, axis=1)
            # scaler = StandardScaler()
            # activation_df_std = scaler.fit_transform(activation_df.values.T)
        case "minmax":
            activation_df_std = minmax_scale(activation_df, axis=1)
            # activation_df_std = pd.DataFrame(
            #     activation_df_std.T,
            #     index=activation_df.index,
            #     columns=activation_df.columns,
            # )
        case None:
            scaler = None
            activation_df_std = activation_df
    activation_df_std = pd.DataFrame(
        activation_df_std, index=activation_df.index, columns=activation_df.columns
    )

    def get_roi(peak_results_row: pd.Series, method: Literal["mask", "clip"] = "mask"):
        # Calculate the start and end indices
        start_index = peak_results_row["start_scan"] - margin
        end_index = peak_results_row["end_scan"] + margin
        # Logger.debug("Start index: %s", start_index)
        # Logger.debug("End index: %s", end_index)
        match method:
            case "mask":
                # Get the original row
                original_row = activation_df_std.loc[peak_results_row["id"]].values
                # Create a new row with all values set to 0
                new_row = np.zeros_like(original_row)
                # Replace the selected part in the new row with the original values
                new_row[start_index : end_index + 1] = original_row[
                    start_index : end_index + 1
                ]
                return new_row[
                    peak_results_row["scan_number_left"] : peak_results_row[
                        "scan_number_right"
                    ]
                    + 1
                ]
            case "clip":
                return activation_df_std.loc[
                    peak_results_row["id"],
                    start_index : end_index + 1,
                ].values

    peaks = peak_results.apply(get_roi, axis=1, method=method)
    return peaks.values


class PeakSelectionModel:
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
                    before = pad_length // 2
                    # after = pad_length - before
                    result[before : before + len(peak)] = peak
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
            Logger.debug("Train indices: %s", self.train_indices[:10])
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
                "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n"
                .format(total, pos, 100 * pos / total)
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
                    "Oversampling positive features, resampled positive features"
                    " shape: %s",
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
                    "Undersampling negative features, resampled negative features"
                    " shape: %s",
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
            case other:
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


def model_tune(config, maxlen, output_bias: float | None = None):
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

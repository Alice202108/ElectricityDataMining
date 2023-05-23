import logging
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

plt.rcParams["savefig.dpi"] = 300  # pixel
plt.rcParams["figure.dpi"] = 300  # resolution
plt.rcParams["figure.figsize"] = [8, 4]  # figure size


def window_truncate(feature_vectors, seq_len, sliding_len=None):
    """Generate time series samples, truncating windows from time-series data with a given sequence length.
    Parameters
    ----------
    feature_vectors: time series data, len(shape)=2, [total_length, feature_num]
    seq_len: sequence length
    sliding_len: size of the sliding window
    """
    sliding_len = seq_len if sliding_len is None else sliding_len
    total_len = feature_vectors.shape[0]
    start_indices = np.asarray(range(total_len // sliding_len)) * sliding_len
    if (
        total_len - start_indices[-1] * sliding_len < seq_len
    ):  # remove the last one if left length is not enough
        start_indices = start_indices[:-1]
    sample_collector = []
    for idx in start_indices:
        sample_collector.append(feature_vectors[idx : idx + seq_len])
    return np.asarray(sample_collector).astype("float32")


def random_mask(vector, artificial_missing_rate):
    """generate indices for random mask"""
    assert len(vector.shape) == 1
    indices = np.where(~np.isnan(vector))[0].tolist()
    indices = np.random.choice(indices, int(len(indices) * artificial_missing_rate))
    return indices


def add_artificial_mask(X, artificial_missing_rate, set_name):
    """Add artificial missing values.
    Parameters
    ----------
    X: feature vectors
    artificial_missing_rate: rate of artificial missing values that are going to be create
    set_name: dataset name, train/val/test
    """
    sample_num, seq_len, feature_num = X.shape
    if set_name == "train":
        # if this is train set, we don't need add artificial missing values right now.
        # If we want to apply MIT during training, dataloader will randomly mask out some values to generate X_hat

        # calculate empirical mean for model GRU-D, refer to paper
        mask = (~np.isnan(X)).astype(np.float32)
        X_filledWith0 = np.nan_to_num(X)
        empirical_mean_for_GRUD = np.sum(mask * X_filledWith0, axis=(0, 1)) / np.sum(
            mask, axis=(0, 1)
        )
        data_dict = {
            "X": X,
            "empirical_mean_for_GRUD": empirical_mean_for_GRUD,
        }
    else:
        # if this is val/test set, then we need to add artificial missing values right now,
        # because we need they are fixed
        X = X.reshape(-1)
        indices_for_holdout = random_mask(X, artificial_missing_rate)
        X_hat = np.copy(X)
        X_hat[indices_for_holdout] = np.nan  # X_hat contains artificial missing values
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        # indicating_mask contains masks indicating artificial missing values
        indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32)

        data_dict = {
            "X": X.reshape([sample_num, seq_len, feature_num]),
            "X_hat": X_hat.reshape([sample_num, seq_len, feature_num]),
            "missing_mask": missing_mask.reshape([sample_num, seq_len, feature_num]),
            "indicating_mask": indicating_mask.reshape(
                [sample_num, seq_len, feature_num]
            ),
        }

    return data_dict


def saving_into_h5(saving_dir, data_dict, classification_dataset):
    """Save data into h5 file.
    Parameters
    ----------
    saving_dir: path of saving dir
    data_dict: data dictionary containing train/val/test sets
    classification_dataset: boolean, if this is a classification dataset
    """

    def save_each_set(handle, name, data):
        single_set = handle.create_group(name)
        if classification_dataset:
            single_set.create_dataset("labels", data=data["labels"].astype(int))
        single_set.create_dataset("X", data=data["X"].astype(np.float32))
        if name in ["val", "test"]:
            single_set.create_dataset("X_hat", data=data["X_hat"].astype(np.float32))
            single_set.create_dataset(
                "missing_mask", data=data["missing_mask"].astype(np.float32)
            )
            single_set.create_dataset(
                "indicating_mask", data=data["indicating_mask"].astype(np.float32)
            )

    saving_path = os.path.join(saving_dir, "datasets.h5")
    with h5py.File(saving_path, "w") as hf:
        hf.create_dataset(
            "empirical_mean_for_GRUD",
            data=data_dict["train"]["empirical_mean_for_GRUD"],
        )
        save_each_set(hf, "train", data_dict["train"])
        save_each_set(hf, "val", data_dict["val"])
        save_each_set(hf, "test", data_dict["test"])


def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_mse_cal(inputs, target, mask):
    """calculate Mean Square Error"""
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(inputs, target, mask):
    """calculate Root Mean Square Error"""
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(inputs, target, mask):
    """calculate Mean Relative Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (
        torch.sum(torch.abs(target * mask)) + 1e-9
    )


def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = metrics.precision_recall_curve(
        y_true=y_test, probas_pred=y_pred
    )
    area = metrics.auc(recalls, precisions)
    return area, precisions, recalls, thresholds


def auc_roc(y_pred, y_test):
    auc = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds


def auc_to_recall(recalls, precisions, recall=0.01):
    precisions_mod = precisions.copy()
    ind = np.where(recalls < recall)[0][0] + 1
    precisions_mod[:ind] = 0
    area = metrics.auc(recalls, precisions_mod)
    return area


def cal_classification_metrics(probabilities, labels, pos_label=1, class_num=1):
    """
    pos_label: The label of the positive class.
    """
    if class_num == 1:
        class_predictions = (probabilities >= 0.5).astype(int)
    elif class_num == 2:
        class_predictions = np.argmax(probabilities, axis=1)
    else:
        assert "args.class_num>2, class need to be specified for precision_recall_fscore_support"
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        labels, class_predictions, pos_label=pos_label, warn_for=()
    )
    precision, recall, f1 = precision[1], recall[1], f1[1]
    precisions, recalls, _ = metrics.precision_recall_curve(
        labels, probabilities[:, -1], pos_label=pos_label
    )
    acc_score = metrics.accuracy_score(labels, class_predictions)
    ROC_AUC, fprs, tprs, thresholds = auc_roc(probabilities[:, -1], labels)
    PR_AUC = metrics.auc(recalls, precisions)
    classification_metrics = {
        "classification_predictions": class_predictions,
        "acc_score": acc_score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precisions": precisions,
        "recalls": recalls,
        "fprs": fprs,
        "tprs": tprs,
        "ROC_AUC": ROC_AUC,
        "PR_AUC": PR_AUC,
    }
    return classification_metrics


def plot_AUCs(
    pdf_file, x_values, y_values, auc_value, title, x_name, y_name, dataset_name
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        x_values,
        y_values,
        ".",
        label=f"{dataset_name}, AUC={auc_value:.3f}",
        rasterized=True,
    )
    l = ax.legend(fontsize=10, loc="lower left")
    l.set_zorder(20)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title, fontsize=12)
    pdf_file.savefig(fig)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise TypeError("Boolean value expected.")


def setup_logger(log_file_path, log_name, mode="a"):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False  # prevent the child logger from propagating log to the root logger (twice), not necessary
    return logger


class Controller:
    def __init__(self, early_stop_patience):
        self.original_early_stop_patience_value = early_stop_patience
        self.early_stop_patience = early_stop_patience
        self.state_dict = {
            # `step` is for training stage
            "train_step": 0,
            # below are for validation stage
            "val_step": 0,
            "epoch": 0,
            "best_imputation_MAE": 1e9,
            "should_stop": False,
            "save_model": False,
        }

    def epoch_num_plus_1(self):
        self.state_dict["epoch"] += 1

    def __call__(self, stage, info=None, logger=None):
        if stage == "train":
            self.state_dict["train_step"] += 1
        else:
            self.state_dict["val_step"] += 1
            self.state_dict["save_model"] = False
            current_imputation_MAE = info["imputation_MAE"]
            imputation_MAE_dropped = False  # flags to decrease early stopping patience

            # update best_loss
            if current_imputation_MAE < self.state_dict["best_imputation_MAE"]:
                logger.info(
                    f"best_imputation_MAE has been updated to {current_imputation_MAE}"
                )
                self.state_dict["best_imputation_MAE"] = current_imputation_MAE
                imputation_MAE_dropped = True
            if imputation_MAE_dropped:
                self.state_dict["save_model"] = True

            if self.state_dict["save_model"]:
                self.early_stop_patience = self.original_early_stop_patience_value
            else:
                # if use early_stopping, then update its patience
                if self.early_stop_patience > 0:
                    self.early_stop_patience -= 1
                elif self.early_stop_patience == 0:
                    logger.info(
                        "early_stop_patience has been exhausted, stop training now"
                    )
                    self.state_dict["should_stop"] = True  # to stop training process
                else:
                    pass  # which means early_stop_patience_value is set as -1, not work

        return self.state_dict


def check_saving_dir_for_model(args, time_now):
    saving_path = os.path.join(args.result_saving_base_dir, args.model_name)
    if not args.test_mode:
        log_saving = os.path.join(saving_path, "logs")
        model_saving = os.path.join(saving_path, "models")
        sub_model_saving = os.path.join(model_saving, time_now)
        [
            os.makedirs(dir_)
            for dir_ in [model_saving, log_saving, sub_model_saving]
            if not os.path.exists(dir_)
        ]
        return sub_model_saving, log_saving
    else:
        log_saving = os.path.join(saving_path, "test_log")
        return None, log_saving


def save_model(model, optimizer, model_state_info, args, saving_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(), # don't save optimizer, considering GANs have 2 optimizers
        "training_step": model_state_info["train_step"],
        "epoch": model_state_info["epoch"],
        "model_state_info": model_state_info,
        "args": args,
    }
    torch.save(checkpoint, saving_path)


def load_model(model, checkpoint_path, logger):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Already restored model from checkpoint: {checkpoint_path}")
    return model


def load_model_saved_with_module(model, checkpoint_path, logger):
    """
    To load models those are trained in parallel and saved with module (need to remove 'module.'
    """
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = dict()
    for k, v in checkpoint["model_state_dict"].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    logger.info(f"Already restored model from checkpoint: {checkpoint_path}")
    return model

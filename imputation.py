import numpy as np
from pypots.imputation import SAITS
from tsdb import pickle_load

dataset = pickle_load("data/elec")

saits = SAITS(
    n_steps=100,
    n_features=37,
    n_layers=2,
    d_model=256,
    d_inner=128,
    n_heads=4,
    d_k=64,
    d_v=64,
    dropout=0.1,
    epochs=500,
    patience=10,
    batch_size=128,
    saving_path="results/elec_saits",
)


train_set = {
    "X": dataset["train_X_intact"],
}

val_set = {
    "X": dataset["val_X_hat"],
    "X_intact": np.nan_to_num(dataset["val_X_intact"]),
    "indicating_mask": dataset["val_indicating_mask"],
}

saits.fit(train_set, val_set)

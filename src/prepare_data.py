"""Build a model-ready DopNet dataset: pre-processed images + hand-crafted features.

Interim script so model work can start before the final data-processing code exists.
Pre-processing constants and the train/test split are copied from wicos.ipynb, so the
outputs line up with the notebook (same SEED, same stratified 20% test set).

Usage:  python src/prepare_data.py
Input:  data/Person */<gesture>/Person?g?s*.csv  (unzipped DopNet archives)
Output: data/processed/wicos_dataset.npz   images, features, labels, split (see bottom of file)
        data/processed/features.csv      same features + metadata, one row per sample
        data/processed/feature_ranking.csv mutual information per feature (training set only)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent  # repo root
DATA_ROOT = ROOT / "data"
OUT_DIR = DATA_ROOT / "processed"

# file names: g1=wave, g2=pinch, g3=swipe, g4=click
GESTURES = ["wave", "pinch", "swipe", "click"]

# Pre-processing constants (same as wicos.ipynb)
DOPPLER_CROP = (200, 600)   # keep Doppler bins 200-599 around zero Doppler (bin 400)
DB_FLOOR = -40              # clip everything below the background noise level
OUT_SHAPE = (128, 128)      # (Doppler, time) input size for the CNN

# Zero Doppler (no motion) after cropping: bin 400 of the original 800
ZERO_BIN = 400 - DOPPLER_CROP[0]

# Envelope = Doppler bin where the cumulative power of a time slice reaches this fraction
ENVELOPE_LEVELS = {"lower": 0.03, "central": 0.50, "upper": 0.97}


# ---------------------------------------------------------------- loading ---
def load_data(root=DATA_ROOT):
    """Load all CSVs (no header row). Returns a list of arrays and a metadata DataFrame."""
    # Sorted so the sample order does not depend on the file system
    paths = sorted(root.glob("Person */*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {root.resolve()}")

    spectrograms, metadata_list = [], []
    for path in paths:
        person, gesture, sample = re.search(r"Person([A-Z])g(\d)s(\d+)\.csv$", path.name).groups()
        spectrograms.append(pd.read_csv(path, header=None).to_numpy(dtype=np.float32))
        gesture_index = int(gesture) - 1
        metadata_list.append({
            "person": person,
            "label": gesture_index,
            "gesture": GESTURES[gesture_index],
            "sample": int(sample),
        })
    return spectrograms, pd.DataFrame(metadata_list)


# ---------------------------------------------------------- pre-processing ---
def crop_and_floor(x, crop=DOPPLER_CROP, floor=DB_FLOOR):
    """Doppler crop and dynamic-range floor; values stay in dB, length stays original."""
    return np.maximum(x[crop[0]:crop[1]], floor)


def preprocess(x, crop=DOPPLER_CROP, floor=DB_FLOOR, out_shape=OUT_SHAPE):
    """Pre-process one spectrogram: Doppler crop, dB floor, scale to [0, 1], resize (as in wicos.ipynb)."""
    x = crop_and_floor(x, crop, floor)
    x = (x - floor) / (0 - floor)
    if out_shape is not None:
        x = zoom(x, (out_shape[0] / x.shape[0], out_shape[1] / x.shape[1]), order=1)
    return x.astype(np.float32)


# ------------------------------------------------------ feature extraction ---
def moments(series, prefix):
    """Mean, std, skewness and kurtosis of a time series (Fioranelli et al., 2016, Sec. 3.2)."""
    return {
        f"{prefix}_mean": np.mean(series),
        f"{prefix}_std": np.std(series),
        f"{prefix}_skew": skew(series),
        f"{prefix}_kurt": kurtosis(series),
    }


def extract_features(x_db):
    """Hand-crafted features from one cropped, floored spectrogram (dB, original length).

    The Doppler axis is expressed in bins relative to zero Doppler, so positive and
    negative values mean motion towards / away from the radar.
    """
    n_doppler, n_time = x_db.shape
    doppler = np.arange(n_doppler) - ZERO_BIN

    # Linear power per time-Doppler cell; the -40 dB floor makes noise ~1e-4 of the peak
    power = 10 ** (x_db / 10)
    slice_power = power.sum(axis=0)

    f = {"duration": n_time}

    # 1. Centroid and bandwidth per time slice (Fioranelli et al., 2016)
    centroid = (doppler[:, None] * power).sum(axis=0) / slice_power
    bandwidth = np.sqrt(((doppler[:, None] - centroid) ** 2 * power).sum(axis=0) / slice_power)
    f |= moments(centroid, "centroid")
    f |= moments(bandwidth, "bandwidth")

    # Oscillation of the centroid (a wave moves back and forth): mean crossings per 100 time bins
    crossings = np.sum(np.diff(np.sign(centroid - centroid.mean())) != 0)
    f["centroid_crossing_rate"] = 100 * crossings / n_time

    # 2. Envelopes from the cumulative power per time slice (after Amin et al., 2019)
    cum = np.cumsum(power, axis=0) / slice_power
    envelopes = {name: doppler[np.argmax(cum >= level, axis=0)] for name, level in ENVELOPE_LEVELS.items()}
    for name, env in envelopes.items():
        f[f"env_{name}_mean"] = env.mean()
        f[f"env_{name}_std"] = env.std()
    f["env_upper_max"] = envelopes["upper"].max()
    f["env_lower_min"] = envelopes["lower"].min()
    f["env_spread_mean"] = (envelopes["upper"] - envelopes["lower"]).mean()
    f["env_spread_max"] = (envelopes["upper"] - envelopes["lower"]).max()

    # 3. Simple energy statistics
    above = x_db > -20
    f["occupancy_20db"] = above.mean()                          # share of cells within 20 dB of the peak
    f["doppler_extent_20db"] = above.any(axis=1).sum()          # Doppler bins that are ever within 20 dB
    f["positive_power_ratio"] = power[doppler > 0].sum() / power[doppler != 0].sum()
    f["mean_level_db"] = x_db.mean()

    # Doppler spectrum (time-averaged): entropy and shape
    spectrum = power.sum(axis=1) / power.sum()
    f["doppler_entropy"] = -np.sum(spectrum * np.log(spectrum))

    # Energy over time: how bursty the gesture is
    energy_t = slice_power / slice_power.max()
    f["time_energy_std"] = energy_t.std()
    f["time_active_ratio"] = (energy_t > 0.5).mean()

    return f


# -------------------------------------------------------------------- main ---
def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading CSVs ...")
    spectrograms, metadata = load_data()
    labels = metadata.label.to_numpy()
    persons = metadata.person.to_numpy()
    print(f"Loaded {len(spectrograms)} spectrograms")

    # Train/test split and 5-fold CV, identical to wicos.ipynb
    idx_train, idx_test = train_test_split(
        np.arange(len(metadata)),
        test_size=0.2,
        stratify=metadata.person + "_" + metadata.gesture,
        random_state=SEED,
    )
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(idx_train, labels[idx_train])
    cv_fold = np.full(len(metadata), -1)          # -1 = test set
    for k, (_, val) in enumerate(kfold):
        cv_fold[idx_train[val]] = k

    # Images for the CNN / MLP
    print("Pre-processing images ...")
    X_img = np.stack([preprocess(x) for x in spectrograms])

    # Features for the classical models (original length, so duration is kept)
    print("Extracting features ...")
    features = pd.DataFrame([extract_features(crop_and_floor(x)) for x in spectrograms])
    assert np.isfinite(features.to_numpy()).all(), "non-finite feature values"

    # Feature ranking on the training set only (guidance; do the real selection inside CV)
    mi = mutual_info_classif(features.iloc[idx_train], labels[idx_train], random_state=SEED)
    ranking = pd.DataFrame({"feature": features.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=False)

    # Save
    np.savez_compressed(
        OUT_DIR / "wicos_dataset.npz",
        X_img=X_img,                                  # (960, 128, 128) float32 in [0, 1]; axes (Doppler, time)
        X_feat=features.to_numpy(np.float32),         # (960, n_features) raw, NOT standardised
        feature_names=features.columns.to_numpy(),
        y=labels,                                     # 0=wave 1=pinch 2=swipe 3=click
        gestures=np.array(GESTURES),
        persons=persons,                              # groups for leave-one-person-out
        sample=metadata["sample"].to_numpy(),
        idx_train=idx_train,
        idx_test=idx_test,
        cv_fold=cv_fold,                              # 0-4 for training samples, -1 for test
    )
    table = pd.concat([metadata, pd.Series(np.where(cv_fold < 0, "test", "train"), name="split"),
                       pd.Series(cv_fold, name="cv_fold"), features], axis=1)
    table.to_csv(OUT_DIR / "features.csv", index=False)
    ranking.to_csv(OUT_DIR / "feature_ranking.csv", index=False)

    print(f"X_img  {X_img.shape}, X_feat {features.shape}, train {len(idx_train)}, test {len(idx_test)}")
    print("Top features by mutual information (training set):")
    print(ranking.head(10).to_string(index=False))
    print(f"Saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

"""Build a model-ready DopNet dataset: pre-processed images + envelope features.

Loads the CSVs, makes the train/test split and CV folds, and calls
    preprocessing.preprocess          -> images for the image models (see preprocessing.py)
    features.extract_features         -> envelope features for the feature models (see features.py)
The split (20% test set, 5-fold CV, both stratified on person and gesture) is explained in
notebooks/1_data_exploration.ipynb.

Usage:  python src/build_dataset.py
Input:  data/Person */<gesture>/Person?g?s*.csv  (unzipped DopNet archives)
Output: data/processed/wicos_dataset.npz   images, features, labels, split (see bottom of file)
        data/processed/features.csv      same features + metadata, one row per sample
        data/processed/feature_ranking.csv mutual information per feature (training set only)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, train_test_split

from features import extract_features
from preprocessing import preprocess

SEED = 42
ROOT = Path(__file__).resolve().parent.parent  # repo root
DATA_ROOT = ROOT / "data"
OUT_DIR = DATA_ROOT / "processed"

# file names: g1=wave, g2=pinch, g3=swipe, g4=click
GESTURES = ["wave", "pinch", "swipe", "click"]


def load_data(root=DATA_ROOT):
    """Load all CSVs (no header row). Returns a list of arrays and a metadata DataFrame."""
    # Sorted so the sample order does not depend on the file system
    paths = sorted(root.glob("Person */*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {root.resolve()}")

    spectrograms, metadata_list = [], []
    for path in paths:
        person, gesture, sample = re.search(r"Person([A-Z])g(\d)s(\d+)\.csv$", path.name).groups()
        spectrograms.append(pd.read_csv(path, header=None).to_numpy())
        gesture_index = int(gesture) - 1
        metadata_list.append({
            "person": person,
            "label": gesture_index,
            "gesture": GESTURES[gesture_index],
            "sample": int(sample),
        })
    return spectrograms, pd.DataFrame(metadata_list)


def split(metadata):
    """20% test set and 5-fold CV on the training set, both stratified on person and gesture.

    Returns idx_train, idx_test (sample indices) and cv_fold (0-4 per training sample, -1 for test).
    """
    strata = (metadata.person + "_" + metadata.gesture).to_numpy()
    idx_train, idx_test = train_test_split(
        np.arange(len(metadata)),
        test_size=0.2,
        stratify=strata,
        random_state=SEED,
    )
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(idx_train, strata[idx_train])
    cv_fold = np.full(len(metadata), -1)          # -1 = test set
    for k, (_, val) in enumerate(kfold):
        cv_fold[idx_train[val]] = k
    return idx_train, idx_test, cv_fold


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading CSVs ...")
    spectrograms, metadata = load_data()
    labels = metadata.label.to_numpy()
    print(f"Loaded {len(spectrograms)} spectrograms")

    idx_train, idx_test, cv_fold = split(metadata)

    # Images for the image models
    print("Pre-processing images ...")
    X_img = preprocess(spectrograms)

    # Features for the feature models (original length, so the duration is kept)
    print("Extracting features ...")
    features = pd.DataFrame([extract_features(x) for x in spectrograms])
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
        persons=metadata.person.to_numpy(),           # groups for leave-one-person-out
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

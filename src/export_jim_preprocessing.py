"""Export the pre-processing of notebooks/preprocessing.ipynb as a model-ready dataset.

The notebook's code cells are run unchanged; its pre-processed images are saved, together with the
27 hand-crafted features of prepare_data.py computed on its noise-free, Doppler-cropped spectrograms.
Labels, persons and the train/test split are taken from wicos_dataset.npz after checking that the
notebook uses the same sample order and split (its noise margin and crop are learned on idx_train).

Usage:  python src/export_jim_preprocessing.py
Input:  notebooks/preprocessing.ipynb, data/Person */<gesture>/*.csv, data/processed/wicos_dataset.npz
Output: data/processed/wicos_dataset_jim.npz   same keys as wicos_dataset.npz, plus noise_margin and doppler_crop
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                   # the notebook's figures are not needed here
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import prepare_data                     # feature extraction

REFERENCE = ROOT / "data" / "processed" / "wicos_dataset.npz"
OUT_FILE = ROOT / "data" / "processed" / "wicos_dataset_jim.npz"


def run_notebook(path):
    """Run all code cells of a notebook (from its own folder) and return the variables it creates."""
    cells = json.load(open(path, encoding="utf-8"))["cells"]
    namespace = {}
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        for cell in cells:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), path.name, "exec"), namespace)
                plt.close("all")
    finally:
        os.chdir(cwd)
    return namespace


def main():
    print("Running notebooks/preprocessing.ipynb ...")
    nb = run_notebook(ROOT / "notebooks" / "preprocessing.ipynb")
    ref = np.load(REFERENCE, allow_pickle=True)

    # Same samples in the same order, and the same train/test split as the models use
    paths = list((ROOT / "data").glob("Person */*/*.csv"))
    assert paths == sorted(paths), "preprocessing.ipynb reads the files in another order than prepare_data.py"
    assert np.array_equal(nb["metadata"].label.to_numpy(), ref["y"])
    assert np.array_equal(nb["metadata"].person.to_numpy(), ref["persons"])
    assert np.array_equal(nb["idx_train"], ref["idx_train"]) and np.array_equal(nb["idx_test"], ref["idx_test"])

    # Features on the noise-free spectrograms inside the notebook's Doppler crop
    crop = nb["DOPPLER_CROP"]
    prepare_data.ZERO_BIN = nb["ZERO_DOPPLER"] - crop[0]      # zero Doppler inside this crop
    features = pd.DataFrame([prepare_data.extract_features(x[crop[0]:crop[1]]) for x in nb["spectrograms_clean"]])
    assert list(features.columns) == list(ref["feature_names"]) and np.isfinite(features.to_numpy()).all()

    np.savez_compressed(
        OUT_FILE,
        X_img=nb["X_img"],                                 # (960, 128, 128) in [0, 1]
        X_feat=features.to_numpy(np.float32),              # (960, 27)
        feature_names=features.columns.to_numpy(),
        y=ref["y"], gestures=ref["gestures"], persons=ref["persons"], sample=ref["sample"],
        idx_train=ref["idx_train"], idx_test=ref["idx_test"], cv_fold=ref["cv_fold"],
        noise_margin=nb["margin_global"], doppler_crop=np.array(crop),
    )
    print(f"Noise margin {nb['margin_global']:.1f} dB, Doppler crop {crop}")
    print(f"X_img {nb['X_img'].shape}, X_feat {features.shape} -> {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

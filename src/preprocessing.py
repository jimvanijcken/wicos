"""Image pre-processing of one DopNet spectrogram (same steps as preprocessing.ipynb).

1. Noise removal: clip every pixel below the noise floor (sample median + NOISE_MARGIN) to the floor.
2. Doppler crop: one fixed window DOPPLER_CROP around zero Doppler (bin 400).
3. Scaling: noise floor -> 0, maximum -> 1.
4. Resize: linear interpolation to OUT_SHAPE (Doppler x time).

NOISE_MARGIN and DOPPLER_CROP are learned on the training samples only (see preprocessing.ipynb);
every step uses only the sample itself, so `preprocess` can be applied to any sample without leakage.
"""
import numpy as np
from scipy.ndimage import zoom

ZERO_DOPPLER = 400          # Doppler bin of zero velocity: the centre of the 800 Doppler bins

# Learned on the training samples in preprocessing.ipynb
NOISE_MARGIN = 7.9          # dB above the sample median that counts as gesture (99th pct of far-bin noise)
DOPPLER_CROP = (52, 748)    # fixed Doppler window around zero Doppler, w = 348
OUT_SHAPE = (128, 128)      # (Doppler, time) input size for the CNN


def remove_noise(x, margin=NOISE_MARGIN):
    """Clip every pixel below the noise floor (sample median + margin) to the floor. Returns (x_clean, floor)."""
    floor = np.median(x) + margin
    return np.maximum(x, floor), floor


def preprocess(x, margin=NOISE_MARGIN, crop=DOPPLER_CROP, out_shape=OUT_SHAPE):
    """Raw spectrogram (dB, Doppler bins x time bins) -> pre-processed image of `out_shape`, values in [0, 1].

    margin: dB above the sample median that counts as gesture (learned from the training noise)
    crop:   fixed Doppler window (start, stop) around zero Doppler (learned from the training gesture energy)
    """
    # 1. Noise removal: clip everything below the noise floor (sample median + margin) to the floor
    x, floor = remove_noise(x, margin)

    # 2. Fixed Doppler crop
    x = x[crop[0]:crop[1]]

    # 3. Scaling: noise floor -> 0, maximum -> 1
    x = (x - floor) / (x.max() - floor)

    # 4. Resize to a fixed shape (Doppler by a fixed factor, time stretched per sample)
    x = zoom(x, (out_shape[0] / x.shape[0], out_shape[1] / x.shape[1]), order=1)
    return np.clip(x, 0, 1).astype(np.float32)

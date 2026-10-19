"""Pre-processing of DopNet micro-Doppler spectrograms.

1. Noise removal: one noise level per clip, the median of the far Doppler bins the hand
   never reaches, applied flat across Doppler. Every pixel not more than MARGIN_DB above
   that level is set to the threshold (noise level + margin).
2. Doppler crop: keep the bins between the two far-bin regions, |bin - 400| < 340.
3. Scaling: threshold -> 0, maximum -> 1.
4. Resize: linear interpolation to OUT_SHAPE (Doppler x time).

Input spectrograms are as returned by build_dataset.load_data(): 2D arrays
(n_doppler=800, n_time) in dB, peak-normalised to 0 dB, zero Doppler at bin 400.
Every step uses only the clip itself, so it can be applied to any clip without leakage.
The constants below are derived on the training clips in notebooks/2_preprocessing.ipynb.
"""

import numpy as np
from scipy.ndimage import zoom

N_DOPPLER = 800
ZERO_DOPPLER = N_DOPPLER // 2

# Far bins: |bin - 400| >= 340, i.e. bins 0-59 and 740-799. Gesture-free except for
# short broadband click stripes, which the median tolerates.
FAR_OFFSET = 340
FAR_BINS = np.r_[0:ZERO_DOPPLER - FAR_OFFSET, ZERO_DOPPLER + FAR_OFFSET:N_DOPPLER]

# Margin above the median noise level. Far-bin noise (stripe frames left out) P99 = 8.3 dB,
# P99.9 = 10.2 dB, so 10 dB leaves ~0.13% noise speckle.
MARGIN_DB = 10.0

# Doppler crop: everything outside is the noise-only far region used for the level.
DOPPLER_CROP = (ZERO_DOPPLER - FAR_OFFSET, ZERO_DOPPLER + FAR_OFFSET)   # bins 60-739

OUT_SHAPE = (128, 128)      # (Doppler, time) input size for the CNN


def noise_level(spec):
    """Per-clip noise level [dB]: median of all far-bin pixels."""
    return float(np.median(spec[FAR_BINS]))


def remove_noise(spec, margin_db=MARGIN_DB):
    """Set every pixel at or below noise level + margin to that threshold. Returns (spec_clean, threshold)."""
    threshold = noise_level(spec) + margin_db
    return np.maximum(spec, threshold), threshold


def preprocess_clip(spec, margin_db=MARGIN_DB, crop=DOPPLER_CROP, out_shape=OUT_SHAPE):
    """Raw spectrogram (dB, 800 x n_time) -> cleaned, cropped image with values in [0, 1].

    out_shape=None skips the resize and keeps the cropped size (680 x n_time).
    """
    # 1. Noise removal (level from the full 800 bins, before cropping)
    x, threshold = remove_noise(spec, margin_db)

    # 2. Doppler crop
    x = x[crop[0]:crop[1]]

    # 3. Scaling: threshold -> 0, maximum -> 1
    peak = x.max()
    if peak <= threshold:               # nothing above the noise: all-zero image
        x = np.zeros_like(x)
    else:
        x = (x - threshold) / (peak - threshold)

    # 4. Resize to a fixed shape (Doppler by a fixed factor, time stretched per clip)
    if out_shape is not None:
        x = zoom(x, (out_shape[0] / x.shape[0], out_shape[1] / x.shape[1]), order=1)
    return np.clip(x, 0, 1).astype(np.float32)


def preprocess(spectrograms, margin_db=MARGIN_DB, crop=DOPPLER_CROP, out_shape=OUT_SHAPE):
    """Apply preprocess_clip to a list of spectrograms.

    Returns an array (n_clips, *out_shape), or a ragged list if out_shape is None.
    """
    out = [preprocess_clip(s, margin_db, crop, out_shape) for s in spectrograms]
    return out if out_shape is None else np.stack(out)

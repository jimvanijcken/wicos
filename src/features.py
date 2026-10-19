"""Envelope features of one DopNet spectrogram (Amin, Zeng and Shan, 2019).

M. G. Amin, Z. Zeng and T. Shan, "Hand Gesture Recognition based on Radar Micro-Doppler Signature
Envelopes", IEEE RadarConf 2019. Sec. II-B defines the positive and negative frequency envelopes,
Sec. IV-A three empirical features. Explained step by step in notebooks/3_features.ipynb.

All features are computed on the gesture power at the original time resolution:

    P(d, t) = 10^(x_clean / 10) - 10^(threshold / 10)

inside the Doppler crop, with x_clean and threshold from `preprocessing.remove_noise` (the same
per-clip noise removal as the images), so pure noise has power 0. The Doppler axis is in bins
relative to zero Doppler: positive means motion towards the radar.

Adaptations to DopNet (the paper uses 1-second windows with one gesture each):
- the envelopes are computed over the gesture span, from the first to the last time bin with at
  least ACTIVE_FRACTION of the power of the strongest time bin;
- each envelope is resampled to ENVELOPE_POINTS points, because the recordings differ in length.
"""
import numpy as np

from preprocessing import DOPPLER_CROP, MARGIN_DB, ZERO_DOPPLER, remove_noise

ACTIVE_FRACTION = 0.05      # gesture span: time bins with >= 5% of the power of the strongest time bin
EFFECTIVE_BANDWIDTH = 0.99  # share of a half's gesture energy inside its effective bandwidth (step 1)
ENVELOPE_POINTS = 16        # samples per envelope after resampling the gesture span to a fixed length

# Threshold factor sigma of eq. 3, fixed for all clips: the median of the paper's per-clip suggestion
# (`paper_sigma`) over both halves of the 768 training clips (IQR 0.003-0.012). TODO: tune sigma as a
# hyperparameter with CV on the training folds (the paper: sigma "can be chosen empirically").
SIGMA = 0.006


def gesture_power(x, margin_db=MARGIN_DB, crop=DOPPLER_CROP):
    """Linear power above the clip's noise threshold inside the Doppler crop, shape (Doppler, time)."""
    x_clean, threshold = remove_noise(x, margin_db)
    return 10 ** (x_clean[crop[0]:crop[1]] / 10) - 10 ** (threshold / 10)


def doppler_axis(crop=DOPPLER_CROP):
    """Doppler bin of every row of the crop, relative to zero Doppler."""
    return np.arange(crop[0], crop[1]) - ZERO_DOPPLER


def active_bins(P):
    """Time bins with gesture: total power >= ACTIVE_FRACTION of the strongest time bin."""
    power_t = P.sum(axis=0)
    return power_t >= ACTIVE_FRACTION * power_t.max()


def gesture_span(P):
    """P restricted to the time bins from the first to the last active time bin (the gesture)."""
    active = np.flatnonzero(active_bins(P))
    return P[:, active[0]:active[-1] + 1]


def halves(P, doppler):
    """Split P into the positive and negative Doppler half. Returns (P_half, speed) per half, with
    speed = |Doppler| of every row; zero Doppler belongs to neither half."""
    pos, neg = doppler > 0, doppler < 0
    return (P[pos], doppler[pos]), (P[neg], -doppler[neg])


def effective_bandwidth(P_half, speed, share=EFFECTIVE_BANDWIDTH):
    """Step 1: row of the half where, counted outward from zero Doppler, `share` of the half's
    gesture energy is reached. Its speed is the maximum Doppler of the gesture on that side."""
    spectrum = P_half.sum(axis=1)
    outward = np.argsort(speed)
    cum = np.cumsum(spectrum[outward]) / spectrum.sum()
    return outward[np.argmax(cum >= share)]


def max_doppler(P_half, speed):
    """Maximum |Doppler| of one half from the effective bandwidth (0 if the half has no energy)."""
    return speed[effective_bandwidth(P_half, speed)] if P_half.sum() > 0 else 0


def paper_sigma(P_half, speed):
    """The paper's suggestion for sigma (Sec. II-B): the ratio of the spectrogram value to the energy E(n)
    at the maximum Doppler (step 1), in the time bin where that Doppler bin is strongest."""
    if P_half.sum() == 0:
        return np.nan
    k_max = effective_bandwidth(P_half, speed)
    n_max = np.argmax(P_half[k_max])
    return P_half[k_max, n_max] / P_half[:, n_max].sum()


def half_envelope(P_half, speed, sigma=SIGMA):
    """Steps 2-4 for one half: |envelope| per time bin.

    E(n) = sum over the half of P(d, n) (eq. 2; P is already a power, so it is not squared again),
    threshold T(n) = sigma * E(n) (eq. 3), and the envelope is the outermost Doppler bin with
    P(d, n) >= T(n). Time bins without energy on this side get envelope 0.
    """
    E = P_half.sum(axis=0)                                    # 2. energy of the half per time bin
    above = (P_half >= sigma * E) & (E > 0)                   # 3-4. cells at or above the threshold
    return np.where(above, speed[:, None], 0).max(axis=0)


def envelopes(P, doppler, sigma=SIGMA):
    """Upper (positive) envelope e_U and lower (negative) envelope e_L per time bin, in Doppler bins."""
    (P_pos, speed_pos), (P_neg, speed_neg) = halves(P, doppler)
    return half_envelope(P_pos, speed_pos, sigma), -half_envelope(P_neg, speed_neg, sigma)


def resample(curve, n=ENVELOPE_POINTS):
    """Linear interpolation of a curve to n equally spaced points over its duration."""
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(curve)), curve)


def extract_features(x, sigma=SIGMA, n=ENVELOPE_POINTS, margin_db=MARGIN_DB, crop=DOPPLER_CROP):
    """Envelope features of one raw spectrogram (dB, 800 x n_time), as a dict.

    env_upper_0..n-1, env_lower_0..n-1: e = [e_U, e_L], each resampled to n points (Sec. II-B)
    length, peak_ratio, doppler_bandwidth: the paper's empirical features T, R and B_w (Sec. IV-A)
    """
    P = gesture_span(gesture_power(x, margin_db, crop))
    doppler = doppler_axis(crop)
    upper, lower = envelopes(P, doppler, sigma)
    (P_pos, speed_pos), (P_neg, speed_neg) = halves(P, doppler)
    f_p, f_n = max_doppler(P_pos, speed_pos), -max_doppler(P_neg, speed_neg)  # step 1, both halves

    f = {f"env_upper_{i}": v for i, v in enumerate(resample(upper, n))}
    f |= {f"env_lower_{i}": v for i, v in enumerate(resample(lower, n))}
    f["length"] = P.shape[1]                                  # T = t_e - t_s [time bins]
    f["peak_ratio"] = abs(f_p) / max(abs(f_n), 1)             # R = |f_p / f_n|
    f["doppler_bandwidth"] = abs(f_p) + abs(f_n)              # B_w = |f_p| + |f_n| [Doppler bins]
    return f

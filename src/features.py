"""Envelope features of one DopNet spectrogram (Amin, Zeng and Shan, 2019).

M. G. Amin, Z. Zeng and T. Shan, "Hand Gesture Recognition based on Radar Micro-Doppler Signature
Envelopes", IEEE RadarConf 2019. Sec. II-B defines the positive and negative frequency envelopes,
Sec. IV-A three empirical features. Explained step by step in notebooks/3_features.ipynb.

All features are computed on the gesture power at the original time resolution:

    P(d, t) = 10^(x_clean / 10) - 10^(threshold / 10)

over all 800 Doppler bins, with x_clean and threshold from `preprocessing.remove_noise` (the same
per-clip noise removal as the images), so pure noise has power 0. Unlike the images, the Doppler
axis is not cropped: nothing is rescaled here, so the outer bins cost nothing. The Doppler axis is
in bins relative to zero Doppler: positive means motion towards the radar.

Adaptations to DopNet (the paper uses 1-second windows with one gesture each):
- the envelopes are computed over the gesture span, from the first to the last time bin with at
  least ACTIVE_FRACTION of the power of the strongest time bin;
- like the paper's fixed 1-second window, each envelope starts at the gesture start and is
  zero-padded to WINDOW time bins (no resampling or averaging), so the gesture length stays in
  the envelope; the few longer gestures are cut at WINDOW.

Additions beyond the paper (compact versions of the same envelopes, for the feature models):
- time-normalised envelopes: each envelope averaged over NORM_POINTS equal segments of the gesture
  span, so gestures of different speed line up; the length is the separate feature `length`;
- summary statistics of |e_U| and |e_L| over the gesture span;
- the length of the whole recording (`duration`), next to the gesture span (`length`).
"""
import numpy as np

from preprocessing import MARGIN_DB, N_DOPPLER, ZERO_DOPPLER, remove_noise

ACTIVE_FRACTION = 0.05      # gesture span: time bins with >= 5% of the power of the strongest time bin
EFFECTIVE_BANDWIDTH = 0.999  # share of a half's gesture energy inside its effective bandwidth (step 1)
WINDOW = 358                # envelope length [time bins]: 99% of the 768 training gesture spans fit
NORM_POINTS = 16            # segments per time-normalised envelope (shortest gesture span: 51 time bins)


def gesture_power(x, margin_db=MARGIN_DB):
    """Linear power above the clip's noise threshold, shape (Doppler, time)."""
    x_clean, threshold = remove_noise(x, margin_db)
    return 10 ** (x_clean / 10) - 10 ** (threshold / 10)


def doppler_axis():
    """Doppler bin of every row, relative to zero Doppler."""
    return np.arange(N_DOPPLER) - ZERO_DOPPLER


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
    """Threshold factor sigma of one half (Sec. II-B, eq. 3): the ratio of the spectrogram value to the
    energy E(n) at the maximum Doppler (step 1), in the time bin where that Doppler bin is strongest.
    The threshold then lies exactly at the maximum Doppler in that time bin."""
    k_max = effective_bandwidth(P_half, speed)
    n_max = np.argmax(P_half[k_max])
    return P_half[k_max, n_max] / P_half[:, n_max].sum()


def half_envelope(P_half, speed):
    """Steps 2-4 for one half: |envelope| per time bin.

    E(n) = sum over the half of P(d, n) (eq. 2; P is already a power, so it is not squared again),
    threshold T(n) = sigma * E(n) with the half's own sigma (eq. 3, `paper_sigma`), and the envelope
    is the outermost Doppler bin with P(d, n) >= T(n). Time bins without energy on this side, and
    halves without any energy, get envelope 0.
    """
    if P_half.sum() == 0:
        return np.zeros(P_half.shape[1], dtype=speed.dtype)
    E = P_half.sum(axis=0)                                    # 2. energy of the half per time bin
    sigma = paper_sigma(P_half, speed)                        # 3. threshold factor of this half
    above = (P_half >= sigma * E) & (E > 0)                   # 3-4. cells at or above the threshold
    return np.where(above, speed[:, None], 0).max(axis=0)


def envelopes(P, doppler):
    """Upper (positive) envelope e_U and lower (negative) envelope e_L per time bin, in Doppler bins."""
    (P_pos, speed_pos), (P_neg, speed_neg) = halves(P, doppler)
    return half_envelope(P_pos, speed_pos), -half_envelope(P_neg, speed_neg)


def fixed_window(curve, n=WINDOW):
    """The curve in a window of n time bins starting at the gesture start: zero-padded or cut."""
    out = np.zeros(n)
    out[:min(n, len(curve))] = curve[:n]
    return out


def time_normalised(curve, n=NORM_POINTS):
    """Mean of the curve over n equal segments of its duration. Averaging (not point sampling)
    lets every time bin count and damps the single-bin noise spikes of the envelopes."""
    return np.array([s.mean() for s in np.array_split(curve, n)])


def envelope_stats(speed, prefix):
    """Summary statistics of one envelope magnitude |e(n)| over the gesture span, as a dict.

    mean, std, median: level and variation of the envelope; p90: robust maximum (a few noise spikes
    do not move it, unlike the max).
    """
    return {
        f"{prefix}_mean": speed.mean(),
        f"{prefix}_std": speed.std(),
        f"{prefix}_median": np.median(speed),
        f"{prefix}_p90": np.percentile(speed, 90),
    }


def extract_features(x, n=WINDOW, margin_db=MARGIN_DB):
    """Envelope features of one raw spectrogram (dB, 800 x n_time), as a dict.

    env_upper_0..n-1, env_lower_0..n-1: e = [e_U, e_L], each in a window of n time bins (Sec. II-B)
    length, peak_ratio, doppler_bandwidth: the paper's empirical features T, R and B_w (Sec. IV-A)
    upper_norm_0..15, lower_norm_0..15: e_U and e_L time-normalised to NORM_POINTS segments
    upper_*, lower_* (mean, std, median, p90): statistics of |e_U| and |e_L|
    duration: length of the whole recording [time bins], before cutting out the gesture span
    """
    P = gesture_span(gesture_power(x, margin_db))
    doppler = doppler_axis()
    upper, lower = envelopes(P, doppler)
    (P_pos, speed_pos), (P_neg, speed_neg) = halves(P, doppler)
    f_p, f_n = max_doppler(P_pos, speed_pos), -max_doppler(P_neg, speed_neg)  # step 1, both halves

    f = {f"env_upper_{i}": v for i, v in enumerate(fixed_window(upper, n))}
    f |= {f"env_lower_{i}": v for i, v in enumerate(fixed_window(lower, n))}
    f["length"] = P.shape[1]                                  # T = t_e - t_s [time bins]
    f["peak_ratio"] = abs(f_p) / max(abs(f_n), 1)             # R = |f_p / f_n|
    f["doppler_bandwidth"] = abs(f_p) + abs(f_n)              # B_w = |f_p| + |f_n| [Doppler bins]

    f |= {f"upper_norm_{i}": v for i, v in enumerate(time_normalised(upper))}
    f |= {f"lower_norm_{i}": v for i, v in enumerate(time_normalised(lower))}
    f |= envelope_stats(upper, "upper")
    f |= envelope_stats(-lower, "lower")                      # |e_L|: speed away from the radar
    f["duration"] = x.shape[1]
    return f

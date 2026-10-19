"""Hand-crafted features of one DopNet spectrogram.

All features are computed on the gesture power at the original time resolution:

    P(d, t) = 10^(x_clean / 10) - 10^(floor / 10)

inside the Doppler crop, with x_clean and floor from `preprocessing.remove_noise` (the same per-sample
noise floor as the images). Noise below the floor gets power 0. Time bins whose total power is below
ACTIVE_FRACTION of the strongest time bin are skipped for the per-time-bin curves: before and after the
gesture they contain only a few noise pixels, whose power-weighted centroid and envelopes jump randomly.

The Doppler axis is in bins relative to zero Doppler, so positive and negative values mean motion
towards / away from the radar.
"""
import numpy as np
from scipy.stats import kurtosis, skew

from preprocessing import DOPPLER_CROP, NOISE_MARGIN, ZERO_DOPPLER, remove_noise

ACTIVE_FRACTION = 0.05      # keep time bins with >= 5% of the power of the strongest time bin
ENVELOPE_LEVELS = {"lower": 0.05, "central": 0.50, "upper": 0.95}  # cumulative power per time bin
EXTENT_LEVELS = (0.01, 0.99)  # Doppler extent: bins between these cumulative shares of the Doppler spectrum


def gesture_power(x, margin=NOISE_MARGIN, crop=DOPPLER_CROP):
    """Linear power above the sample's noise floor inside the Doppler crop, shape (Doppler, time)."""
    x_clean, floor = remove_noise(x, margin)
    return 10 ** (x_clean[crop[0]:crop[1]] / 10) - 10 ** (floor / 10)


def doppler_axis(crop=DOPPLER_CROP):
    """Doppler bin of every row of the crop, relative to zero Doppler."""
    return np.arange(crop[0], crop[1]) - ZERO_DOPPLER


def active_bins(P):
    """Time bins with gesture: total power >= ACTIVE_FRACTION of the strongest time bin."""
    power_t = P.sum(axis=0)
    return power_t >= ACTIVE_FRACTION * power_t.max()


def centroid_bandwidth(P, doppler):
    """Doppler centroid f_c(t) and bandwidth B(t) per time bin (Fioranelli et al., 2016)."""
    slice_power = P.sum(axis=0)
    centroid = doppler @ P / slice_power
    bandwidth = np.sqrt(((doppler[:, None] - centroid) ** 2 * P).sum(axis=0) / slice_power)
    return centroid, bandwidth


def envelopes(P, doppler):
    """Lower, central and upper envelope per time bin (after Amin et al., 2019), as a dict."""
    cum = np.cumsum(P, axis=0) / P.sum(axis=0)
    return {name: doppler[np.argmax(cum >= level, axis=0)] for name, level in ENVELOPE_LEVELS.items()}


def moments(series, prefix):
    """Mean, std, skewness and kurtosis of a time series (Fioranelli et al., 2016, Sec. 3.2)."""
    return {
        f"{prefix}_mean": np.mean(series),
        f"{prefix}_std": np.std(series),
        f"{prefix}_skew": skew(series),
        f"{prefix}_kurt": kurtosis(series),
    }


def extract_features(x, margin=NOISE_MARGIN, crop=DOPPLER_CROP):
    """Hand-crafted features from one raw spectrogram (dB, Doppler bins x time bins), as a dict."""
    P_all = gesture_power(x, margin, crop)
    doppler = doppler_axis(crop)

    # Only the time bins with gesture
    active = active_bins(P_all)
    P = P_all[:, active]
    slice_power = P.sum(axis=0)

    f = {
        "duration": P_all.shape[1],          # recording length [time bins]
        "active_duration": active.sum(),     # time bins with gesture
    }

    # 1. Centroid and bandwidth per time bin (Fioranelli et al., 2016)
    centroid, bandwidth = centroid_bandwidth(P, doppler)
    f |= moments(centroid, "centroid")
    f |= moments(bandwidth, "bandwidth")

    # Oscillation of the centroid (a wave moves back and forth): mean crossings per 100 active time bins
    crossings = np.sum(np.diff(np.sign(centroid - centroid.mean())) != 0)
    f["centroid_crossing_rate"] = 100 * crossings / active.sum()

    # 2. Envelopes from the cumulative power per time bin (after Amin et al., 2019)
    env = envelopes(P, doppler)
    for name, e in env.items():
        f[f"env_{name}_mean"] = e.mean()
        f[f"env_{name}_std"] = e.std()
    f["env_upper_max"] = env["upper"].max()
    f["env_lower_min"] = env["lower"].min()
    f["env_spread_mean"] = (env["upper"] - env["lower"]).mean()
    f["env_spread_max"] = (env["upper"] - env["lower"]).max()

    # 3. Doppler spectrum (time-averaged gesture power): extent, direction, entropy
    spectrum = P.sum(axis=1) / P.sum()
    cum_spectrum = np.cumsum(spectrum)
    lo, hi = (np.argmax(cum_spectrum >= q) for q in EXTENT_LEVELS)
    f["doppler_extent"] = hi - lo + 1
    f["positive_power_ratio"] = P[doppler > 0].sum() / P[doppler != 0].sum()
    nonzero = spectrum[spectrum > 0]
    f["doppler_entropy"] = -np.sum(nonzero * np.log(nonzero))
    f["occupancy"] = (P > 0).mean()          # share of time-Doppler cells above the noise floor (active bins)

    # 4. Energy over time (active bins): how bursty the gesture is
    energy_t = slice_power / slice_power.max()
    f["time_energy_std"] = energy_t.std()
    f["time_half_power_ratio"] = (energy_t > 0.5).mean()

    return f


# ------------------------------------------------------ Amin et al. (2019) envelopes ---
# Positive and negative frequency envelopes as in Amin, Zeng and Shan, "Hand Gesture Recognition based
# on Radar Micro-Doppler Signature Envelopes", RadarConf 2019, Sec. II-B. Explained in envelope_features.ipynb.
EFFECTIVE_BANDWIDTH = 0.99  # share of a half's gesture energy inside its effective bandwidth (step 1)
ENVELOPE_POINTS = 16        # samples per envelope after resampling the gesture to a fixed length


def gesture_span(P):
    """P restricted to the time bins from the first to the last active time bin (the gesture)."""
    active = np.flatnonzero(active_bins(P))
    return P[:, active[0]:active[-1] + 1]


def effective_bandwidth(P_half, speed, share=EFFECTIVE_BANDWIDTH):
    """Step 1: index of the Doppler bin of the half (|Doppler| = speed) where, counted outward from zero
    Doppler, `share` of the half's gesture energy is reached: the maximum Doppler of the gesture."""
    spectrum = P_half.sum(axis=1)
    outward = np.argsort(speed)
    cum = np.cumsum(spectrum[outward]) / spectrum.sum()
    return outward[np.argmax(cum >= share)]


def half_envelope(P_half, speed, sigma=None):
    """Steps 2-4 for one half (positive or negative Doppler). Returns (|envelope| per time bin, sigma).

    E(n) = sum over the half of P(d, n), threshold T(n) = sigma * E(n), and the envelope is the outermost
    Doppler bin with P(d, n) >= T(n). Without a given sigma it follows the paper's suggestion: the ratio
    T / E is fixed at the time and Doppler bin where the maximum Doppler (step 1) is reached.
    """
    E = P_half.sum(axis=0)                                    # 2. energy of the half per time bin
    if E.max() == 0:                                          # no gesture energy on this side
        return np.zeros(P_half.shape[1]), np.nan
    if sigma is None:                                         # 3. sigma from the maximum Doppler
        k_max = effective_bandwidth(P_half, speed)
        n_max = np.argmax(P_half[k_max])
        sigma = P_half[k_max, n_max] / E[n_max]
    above = (P_half >= sigma * E) & (E > 0)                   # 4. cells at or above the threshold
    return np.where(above, speed[:, None], 0).max(axis=0), sigma


def amin_envelopes(P, doppler, sigma=None):
    """Upper (positive) and lower (negative) frequency envelope per time bin, and their sigmas."""
    pos, neg = doppler > 0, doppler < 0
    upper, sigma_u = half_envelope(P[pos], doppler[pos], sigma)
    lower, sigma_l = half_envelope(P[neg], -doppler[neg], sigma)
    return upper, -lower, sigma_u, sigma_l


def resample(curve, n=ENVELOPE_POINTS):
    """Linear interpolation of a curve to n equally spaced points over its duration."""
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(curve)), curve)


def amin_features(x, sigma=None, n=ENVELOPE_POINTS, margin=NOISE_MARGIN, crop=DOPPLER_CROP):
    """Envelope feature vector e = [e_U, e_L] (each resampled to n points) and the paper's three
    empirical features (Sec. IV-A) of one raw spectrogram. Returns (vector, dict)."""
    P = gesture_span(gesture_power(x, margin, crop))
    upper, lower, _, _ = amin_envelopes(P, doppler_axis(crop), sigma)
    f_p, f_n = upper.max(), lower.min()                       # maximum positive / negative Doppler
    empirical = {
        "length": P.shape[1],                                 # T = t_e - t_s
        "peak_ratio": abs(f_p) / max(abs(f_n), 1),            # R = |f_p / f_n|
        "bandwidth": abs(f_p) + abs(f_n),                     # B_w = |f_p| + |f_n|
    }
    return np.concatenate([resample(upper, n), resample(lower, n)]), empirical

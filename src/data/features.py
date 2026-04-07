import numpy as np
import pandas as pd
from scipy.signal import welch


def apply_causal_smoothing(X, window_size=5):
    kernel = np.ones(window_size) / window_size
    X_smoothed = np.zeros_like(X)
    for i in range(X.shape[1]):
        conv_res = np.convolve(X[:, i], kernel, mode='full')[:X.shape[0]]
        for j in range(window_size - 1):
            conv_res[j] = np.mean(X[:j+1, i])
        X_smoothed[:, i] = conv_res
    return X_smoothed


def extract_concept_features(raw_eeg, eeg_indices, beh_df, fs_eeg=1000, raw_emg=None):
    n_samples = len(beh_df)
    if raw_eeg is None or eeg_indices is None:
        delta_powers = theta_powers = emg_powers = np.zeros(n_samples, dtype=np.float32)
    else:
        half_window = int(1.0 * fs_eeg)
        delta_powers = np.full(n_samples, np.nan, dtype=np.float32)
        theta_powers = np.full(n_samples, np.nan, dtype=np.float32)
        emg_powers   = np.full(n_samples, np.nan, dtype=np.float32)

        for i, pos in enumerate(eeg_indices):
            start, end = pos - half_window, pos + half_window
            if start < 0 or end >= len(raw_eeg): continue

            segment_eeg = raw_eeg[start:end] - np.mean(raw_eeg[start:end])
            f, psd = welch(segment_eeg, fs=fs_eeg, nperseg=min(2 * half_window, 2048))
            delta_mask, theta_mask = (f >= 0.5) & (f <= 4.0), (f >= 4.0) & (f <= 8.0)
            delta_powers[i] = np.trapz(psd[delta_mask], f[delta_mask]) if np.any(delta_mask) else 0.0
            theta_powers[i] = np.trapz(psd[theta_mask], f[theta_mask]) if np.any(theta_mask) else 0.0

            if raw_emg is not None and end < len(raw_emg):
                emg_powers[i] = np.var(raw_emg[start:end])

        df_temp = pd.DataFrame({'delta': delta_powers, 'theta': theta_powers, 'emg': emg_powers}).ffill().fillna(0.0)
        delta_powers, theta_powers, emg_powers = df_temp['delta'].values, df_temp['theta'].values, df_temp['emg'].values

    ratio_dt = delta_powers / (theta_powers + 1e-8)
    behavior_features = np.column_stack([beh_df[col].values for col in ['motor','locomotion','grooming','nestgrooming','nestactivity','rearing','drinking','eating','nesting']])
    return np.column_stack([delta_powers, theta_powers, ratio_dt, emg_powers, behavior_features])


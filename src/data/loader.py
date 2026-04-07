import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
from sklearn.preprocessing import LabelEncoder
from pymatreader import read_mat

def get_parts_for_mouse(base_path, mouse_id):
    """获取指定小鼠的原始数据分段路径"""
    parts, part_idx = [], 1
    while True:
        part_path = os.path.join(base_path, mouse_id, f'part{part_idx}')
        if os.path.exists(part_path):
            parts.append(part_path)
            part_idx += 1
        else:
            break
    if not parts:
        fallback = os.path.join(base_path, mouse_id)
        if os.path.exists(fallback): parts = [fallback]
    return parts

def load_data(part_path, handle_eeg_missing='truncate', fs_calcium=1.92):
    """加载 Mat 格式钙成像、EEG 及行为学 CSV 标签数据，统一采样率"""
    calcium_mat = sio.loadmat(os.path.join(part_path, 'Calcium imaging_Trace.mat'))
    dff, mask = calcium_mat['dff'], calcium_mat['mask'].flatten().astype(bool)

    beh_df = pd.read_csv(os.path.join(part_path, 'Behavior recording_Label.csv'))

    GLOBAL_STAGE_MAP = {
        'MA': 0, 'Microarousal': 0, 'microarousal': 0,
        'NR': 1, 'NREM': 1, 'nrem': 1,
        'R':  2, 'REM': 2, 'rem': 2,
        'W':  3, 'Wake': 3, 'wake': 3
    }
    
    clean_series = beh_df['sleep_stage'].astype(str).str.strip().replace(['', 'nan', 'NaN'], pd.NA)
    
    sleep_stage = clean_series.ffill().bfill().map(GLOBAL_STAGE_MAP).fillna(3).values.astype(int)
    stage_names_local = ['Microarousal', 'NREM', 'REM', 'Wake']

    beh_cols = ['motor','locomotion','grooming','nestgrooming','nestactivity','rearing','drinking','eating','nesting']
    beh_data = beh_df[beh_cols].values.astype(np.float32)

    eeg_file = os.path.join(part_path, 'EEGEMG recording_Filtered.mat')
    if os.path.exists(eeg_file):
        inner = read_mat(eeg_file)['filteredEEG']
        raw_eeg, raw_emg, fs, eeg_available = inner['EEG1'].flatten(), inner['EMG'].flatten(), 1000, True
    else:
        raw_eeg, raw_emg, fs, eeg_available = None, None, None, False

    valid_idx = mask
    X, y_sleep, y_beh = dff[:, valid_idx].T, sleep_stage[valid_idx], beh_data[valid_idx]

    if eeg_available:
        eeg_indices = np.linspace(0, len(raw_eeg) - 1, X.shape[0]).astype(int)
        last_valid_time_idx = np.searchsorted(eeg_indices, len(raw_eeg) - 1, side='right')
        if last_valid_time_idx < X.shape[0]:
            X, y_sleep, y_beh = X[:last_valid_time_idx], y_sleep[:last_valid_time_idx], y_beh[:last_valid_time_idx]
            eeg_indices = eeg_indices[:last_valid_time_idx]
    else:
        eeg_indices = None

    fs_original = 0.96 if os.path.basename(os.path.dirname(part_path)) == '#5' else 1.92
    if fs_original != fs_calcium:
        ratio = fs_calcium / fs_original
        new_len = int(X.shape[0] * ratio)
        X_resampled = np.zeros((new_len, X.shape[1]), dtype=X.dtype)
        for i in range(X.shape[1]): X_resampled[:, i] = signal.resample(X[:, i], new_len)
        X = X_resampled
        y_sleep = np.repeat(y_sleep, int(ratio))
        y_beh = np.repeat(y_beh, int(ratio), axis=0)
        if eeg_available:
            eeg_indices = np.clip(np.linspace(0, len(eeg_indices) - 1, new_len).astype(int), 0, len(raw_eeg) - 1)

    return X, y_sleep, y_beh, beh_cols, stage_names_local, raw_eeg, raw_emg, fs, eeg_indices
import numpy as np

def create_sliding_windows(X, y, C, window_size):
    N = len(X)
    if N < window_size: return None, None, None
    shape_X = (N - window_size + 1, window_size, X.shape[1])
    strides_X = (X.strides[0], X.strides[0], X.strides[1])
    X_seq = np.lib.stride_tricks.as_strided(X, shape=shape_X, strides=strides_X)

    shape_C = (N - window_size + 1, window_size, C.shape[1])
    strides_C = (C.strides[0], C.strides[0], C.strides[1])
    C_seq = np.lib.stride_tricks.as_strided(C, shape=shape_C, strides=strides_C)
    return X_seq, y[window_size - 1:], C_seq

def build_future_dataset(parts, time_lag):
    C_seq_list, Y_fut_list, Y_curr_list = [], [], []
    for p in parts:
        if p.get('C_seq') is None: continue
        valid_len = len(p['C_seq']) - time_lag
        if valid_len > 0:
            C_seq_list.append(p['C_seq'][:valid_len])
            Y_curr_list.append(p['Y_seq'][:valid_len])
            Y_fut_list.append(p['Y_seq'][time_lag:])
    if not C_seq_list: return None, None, None
    return np.vstack(C_seq_list), np.concatenate(Y_fut_list), np.concatenate(Y_curr_list)
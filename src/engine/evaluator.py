import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc
from collections import Counter, defaultdict


# Post-processing & Calibration
def moving_avg_probs(probs, k=3):
    kernel = np.ones(k) / k
    if probs.ndim == 1:
        return np.convolve(probs, kernel, mode='same')
    else:
        smoothed = np.zeros_like(probs)
        for c in range(probs.shape[1]):
            smoothed[:, c] = np.convolve(probs[:, c], kernel, mode='same')
        return smoothed

def calculate_ece(probs, labels, n_bins=10):
    if hasattr(probs, 'detach'):
        probs = probs.detach().cpu().numpy()
    if hasattr(labels, 'detach'):
        labels = labels.detach().cpu().numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def temporal_filter(dynamics_score, threshold, k=3):
    mask = (dynamics_score > threshold).float()
    if mask.dim() == 1:
        mask_3d = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 2:
        mask_3d = mask.unsqueeze(1)

    kernel = torch.ones(1, 1, k, device=dynamics_score.device)
    conv = F.conv1d(mask_3d, kernel, padding=k-1)
    trigger = (conv[..., :mask_3d.shape[-1]] >= (k - 1e-5)).float()
    trigger[..., :k-1] = 0
    return trigger.view_as(dynamics_score)

def apply_cooldown(trigger_1d, cooldown=5):
    assert trigger_1d.ndim == 1, "1D"
    if isinstance(trigger_1d, torch.Tensor):
        final = trigger_1d.clone()
    else:
        final = trigger_1d.copy()

    last_idx = -cooldown
    for t in range(len(trigger_1d)):
        if trigger_1d[t] == 1:
            if (t - last_idx) < cooldown:
                final[t] = 0
            else:
                last_idx = t
    return final

# Evaluation & Statistics
def calculate_transition_metrics(y_true_trans, y_pred_trans, y_prob_trans, pre_frames=10, post_frames=3, fs=1.92):
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_trans, y_prob_trans)
    auprc = auc(recall_curve, precision_curve)

    true_trans_indices = np.where(y_true_trans == 1)[0]
    pred_trans_indices = np.where(y_pred_trans == 1)[0]

    hits = 0
    lead_times = []
    valid_alarms = set()

    for idx in true_trans_indices:
        start_idx = max(0, idx - pre_frames)
        end_idx = min(len(y_pred_trans), idx + post_frames + 1)
        window_preds = y_pred_trans[start_idx : end_idx]
        
        if np.any(window_preds == 1):
            hits += 1
            alarm_indices_in_window = np.where(window_preds == 1)[0]
            first_alarm_idx_absolute = start_idx + alarm_indices_in_window[0]
            lead_time_sec = (idx - first_alarm_idx_absolute) / fs
            
            if lead_time_sec >= 0: 
                lead_times.append(lead_time_sec)
            
            for a_idx in alarm_indices_in_window:
                valid_alarms.add(start_idx + a_idx)

    event_recall = hits / len(true_trans_indices) if len(true_trans_indices) > 0 else 0
    total_alarms = len(pred_trans_indices)
    valid_alarm_count = len(valid_alarms)
    
    precision_val = valid_alarm_count / total_alarms if total_alarms > 0 else 0
    avg_lead_time = np.mean(lead_times) if len(lead_times) > 0 else 0
    
    return auprc, precision_val, event_recall, avg_lead_time

def compute_transition_recall(y_true_fut, y_pred, transition_mask):
    mask = np.asarray(transition_mask, dtype=bool)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.asarray(y_true_fut)[mask] == np.asarray(y_pred)[mask])

# --- Baseline 1 ---
def build_transition_prior(y_curr_train, y_fut_train):
    counts = defaultdict(lambda: defaultdict(int))
    y_curr_train = np.asarray(y_curr_train)
    y_fut_train = np.asarray(y_fut_train)
    for cur, fut in zip(y_curr_train, y_fut_train):
        counts[cur][fut] += 1
    probs = {}
    for cur, d in counts.items():
        total = sum(d.values())
        probs[cur] = {k: v / total for k, v in d.items()}
    return probs

def predict_transition_prior(y_test_curr, trans_prob):
    preds = []
    for cur in y_test_curr:
        if cur in trans_prob:
            best_fut = max(trans_prob[cur].items(), key=lambda x: x[1])[0]
            preds.append(best_fut)
        else:
            preds.append(cur)
    return np.array(preds)

# --- Baseline 2 ---
def majority_baseline(y_train_fut, length):
    counts = Counter(y_train_fut)
    majority_class = counts.most_common(1)[0][0]
    return np.full(length, majority_class)

# --- Baseline 3 ---
def temporal_smoothing_baseline(y_test_curr):
    return y_test_curr


def block_permutation_test(y_true, y_pred, mask, n_permutations=1000, block_size=10):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0: return {"p": 1.0, "z": 0.0}
    actual_score = np.mean(np.asarray(y_true)[mask] == np.asarray(y_pred)[mask])
    
    y_pred_arr = np.asarray(y_pred)
    n = len(y_pred_arr)
    perm_scores = []
    
    for _ in range(n_permutations):
        num_blocks = n // block_size
        blocks = np.array_split(y_pred_arr[:num_blocks*block_size], num_blocks)
        np.random.shuffle(blocks)
        permuted_pred = np.concatenate(blocks)
        if len(permuted_pred) < n:
            permuted_pred = np.concatenate([permuted_pred, y_pred_arr[num_blocks*block_size:]])
            
        score = np.mean(np.asarray(y_true)[mask] == permuted_pred[mask])
        perm_scores.append(score)
        
    perm_scores = np.array(perm_scores)
    p_val = np.mean(perm_scores >= actual_score)
    z_score = (actual_score - np.mean(perm_scores)) / (np.std(perm_scores) + 1e-9)
    return {"p": p_val, "z": z_score}

def bootstrap_ci(y_true, y_pred, mask, n_iterations=1000, alpha=0.05):
    """95% Confidence Interval"""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0: return {"ci_lower": np.nan, "ci_upper": np.nan}
    
    y_t = np.asarray(y_true)[mask]
    y_p = np.asarray(y_pred)[mask]
    n = len(y_t)
    
    scores = []
    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        scores.append(np.mean(y_t[indices] == y_p[indices]))
        
    scores = np.sort(scores)
    lower = scores[int((alpha/2) * n_iterations)]
    upper = scores[int((1 - alpha/2) * n_iterations)]
    return {"ci_lower": lower, "ci_upper": upper}

def full_evaluation(y_train_curr, y_train_fut, y_test_curr, y_test_fut, y_pred_model, transition_mask):
    results = {}
    results["model"] = compute_transition_recall(y_test_fut, y_pred_model, transition_mask)
    
    trans_prob = build_transition_prior(y_train_curr, y_train_fut)
    results["baseline_transition"] = compute_transition_recall(
        y_test_fut, predict_transition_prior(y_test_curr, trans_prob), transition_mask
    )
    results["baseline_majority"] = compute_transition_recall(
        y_test_fut, majority_baseline(y_train_fut, len(y_test_fut)), transition_mask
    )
    results["baseline_temporal"] = compute_transition_recall(
        y_test_fut, temporal_smoothing_baseline(y_test_curr), transition_mask
    )
    
    perm = block_permutation_test(y_test_fut, y_pred_model, transition_mask)
    ci = bootstrap_ci(y_test_fut, y_pred_model, transition_mask)
    
    return {"scores": results, "permutation": perm, "bootstrap": ci}
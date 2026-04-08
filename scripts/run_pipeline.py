import os
import yaml
import torch
import numpy as np
import pandas as pd
import joblib
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.common import seed_everything
from src.data.loader import load_data
from src.data.features import extract_concept_features
from src.models.networks import DualMechanismNet
from src.models.metacognition import CalibratedMetacognitiveLoop

def create_sliding_windows_test(features, seq_len=5):
    N = len(features)
    if N < seq_len:
        raise ValueError("Sequence too short.")
    shape_X = (N - seq_len + 1, seq_len, features.shape[1])
    strides_X = (features.strides[0], features.strides[0], features.strides[1])
    X_seq = np.lib.stride_tricks.as_strided(features, shape=shape_X, strides=strides_X)
    return torch.tensor(X_seq, dtype=torch.float32)

def main():
    print("=" * 65)
    print("[INFO] DynaMeta: End-to-End Metacognitive Pipeline & MAM Dashboard")
    print("=" * 65)
    
    print("[INFO] Loading system configurations...")
    config_path = "configs/default_config.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file missing at: {config_path}")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    seed_everything(config.get('training', {}).get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_names_config = config.get('labels', {}).get('stage_names', ['Microarousal', 'NREM', 'REM', 'Wake'])

    test_slice_dir = "data/raw/#5/part1" 
    print(f"[INFO] Mounting raw sandbox data from: {test_slice_dir}")
    
    if not os.path.exists(test_slice_dir):
        print(f"[ERROR] Data directory not found. Please verify the path.")
        return

    fs_calcium = config.get('signal', {}).get('fs_calcium', 0.96)
    
    (X, y_sleep, y_beh, beh_cols, _, 
     raw_eeg, raw_emg, fs_eeg, eeg_indices) = load_data(
        part_path=test_slice_dir, 
        fs_calcium=fs_calcium
    )
    
    print("[INFO] Extracting multimodal physiological features (PSD, EMG variance)...")
    beh_df = pd.DataFrame(y_beh, columns=beh_cols)
    concept_features = extract_concept_features(
        raw_eeg, eeg_indices, beh_df, fs_eeg=fs_eeg, raw_emg=raw_emg
    )
    
    scaler_path = "checkpoints/feature_scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        concept_features = scaler.transform(concept_features)
        print("[INFO] Applied fitted RobustScaler to pipeline features successfully.")
    else:
        print("[WARNING] Scaler not found at checkpoints/feature_scaler.pkl. Proceeding without normalization.")
    
    seq_len = config.get('signal', {}).get('seq_len_encoder', 5)
    num_concepts = concept_features.shape[1]
    
    X_test_tensor = create_sliding_windows_test(concept_features, seq_len=seq_len).to(device)
    print(f"[INFO] Generated test sequence tensor: {X_test_tensor.shape}")

    # 4. Model Setup
    print("[INFO] Initializing DualMechanismNet...")
    hidden_dim = config.get('model', {}).get('hidden_dim', 64)
    num_classes = config.get('model', {}).get('num_classes', 4)
    
    model = DualMechanismNet(
        input_dim=num_concepts, 
        hidden_dim=hidden_dim,
        num_classes=num_classes
    ).to(device)
    
    weight_path = "checkpoints/dynameta_best.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"[INFO] Loaded trained weights from: {weight_path}")
    else:
        print("[WARNING] Trained weights not found. Using randomly initialized model.")
    
    print("[INFO] Computing data-adaptive MAM thresholds from test distribution...")
    
    proto_path = "checkpoints/prototypes.pth"
    if os.path.exists(proto_path):
        real_prototypes = torch.load(proto_path, map_location=device)
        print("[INFO] Loaded real prototypes from checkpoints.")
    else:
        print("[WARNING] Prototypes not found. Using random initialized ones.")
        real_prototypes = torch.randn(num_classes, hidden_dim).to(device)
        
    dummy_anchor_dict = {i: torch.zeros(num_concepts).to(device) for i in range(num_classes)}
    
    all_entropy, all_dist, all_dyn = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), 512):
            bx = X_test_tensor[i:i+512]
            logits_s_open, _, emb_open, _ = model(bx, return_emb=True)
            
            probs_s_open = F.softmax(logits_s_open[:, -1, :], dim=-1)
            ent = -torch.sum(probs_s_open * torch.log(probs_s_open + 1e-9), dim=-1)
            
            emb_norm = F.normalize(emb_open, p=2, dim=1)
            cos_sim = torch.matmul(emb_norm, real_prototypes.T)
            dist = 1.0 - torch.max(cos_sim, dim=1)[0]
            
            delta_c = bx[:, -1, :] - bx[:, -5:, :].mean(dim=1)
            dyn = torch.norm(delta_c, p=2, dim=-1)
            
            all_entropy.append(ent.cpu().numpy())
            all_dist.append(dist.cpu().numpy())
            all_dyn.append(dyn.cpu().numpy())

    ent_np = np.concatenate(all_entropy)
    dist_np = np.concatenate(all_dist)
    dyn_np = np.concatenate(all_dyn)

    T_ENTROPY = np.percentile(ent_np, 85)
    T_DYN = np.percentile(dyn_np, 95)    
    T_DIST = np.percentile(dist_np, 99)   

    print(f"  > t_entropy (85th) set to: {T_ENTROPY:.4f}")
    print(f"  > t_dyn     (95th) set to: {T_DYN:.4f}")
    print(f"  > t_dist    (99th) set to: {T_DIST:.4f}")

    print("[INFO] Initializing CalibratedMetacognitiveLoop...")
    meta_loop_model = CalibratedMetacognitiveLoop(
        base_model=model,
        anchor_dict_state=dummy_anchor_dict,
        prototypes=real_prototypes,
        temperature=1.0, # Demo 环境默认 1.0
        entropy_s_thresh=T_ENTROPY,
        proto_dist_thresh=T_DIST,
        dyn_thresh=T_DYN,
        mask_lr=1.0,
        max_iters=3
    ).to(device)

    print("[INFO] Executing forward inference and MAM routing...")
    mam_stats = {"defer": 0, "expand": 0, "mask": 0, "pass": 0}
    
    meta_loop_model.eval()
    for i in tqdm(range(0, len(X_test_tensor), 256), desc="Dual-system streaming"):
        bx = X_test_tensor[i:i+256]
        
        final_probs_s, probs_t, dyn_score, final_mask, routes = meta_loop_model(bx, apply_intervention=True)
        
        mam_stats["defer"] += routes["defer"].sum().item()
        mam_stats["expand"] += routes["expand"].sum().item()
        mam_stats["mask"] += routes["mask"].sum().item()
        mam_stats["pass"] += routes["pass"].sum().item()

    total_samples = len(X_test_tensor)
    print("\n" + "=" * 65)
    print(f"  · Strategy D [Fast Pass]          : {mam_stats['pass']:>5d} samples ({mam_stats['pass']/total_samples*100:>5.2f}%)")
    print(f"  · Strategy A [IG Feature Mask]    : {mam_stats['mask']:>5d} samples ({mam_stats['mask']/total_samples*100:>5.2f}%)")
    print(f"  · Strategy B [Transition Expand]  : {mam_stats['expand']:>5d} samples ({mam_stats['expand']/total_samples*100:>5.2f}%)")
    print(f"  · Strategy C [Cognitive Deferral] : {mam_stats['defer']:>5d} samples ({mam_stats['defer']/total_samples*100:>5.2f}%)")
    print("-" * 65)
    print("[SUCCESS] Pipeline streaming finished.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
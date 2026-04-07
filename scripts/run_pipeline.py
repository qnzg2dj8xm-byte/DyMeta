import os
import yaml
import torch
import pandas as pd
import joblib

from src.utils.common import seed_everything
from src.data.loader import load_data
from src.data.features import extract_concept_features
from src.models.networks import DualMechanismNet
from src.models.metacognition import TemperatureScaler, MetacognitiveIntervention

def main():
    print("=" * 65)
    print("[INFO] DynaMeta: End-to-End Metacognitive Pipeline Test")
    print("=" * 65)
    
    # 1. Configuration & Initialization
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

    # 2. Data Loading
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
    
    # 3. Feature Extraction & Normalization
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
    dummy_input = torch.tensor(concept_features[-seq_len:]).unsqueeze(0).float().to(device)

    # 4. Model & Metacognitive Intervention Setup
    print("[INFO] Initializing DualMechanismNet and Sys2 Intervener...")
    model = DualMechanismNet(
        input_dim=concept_features.shape[1], 
        hidden_dim=config.get('model', {}).get('hidden_dim', 64),
        num_classes=config.get('model', {}).get('num_classes', 4)
    ).to(device)
    
    weight_path = "checkpoints/dynameta_best.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"[INFO] Loaded trained weights from: {weight_path}")
    else:
        print("[WARNING] Trained weights not found. Using randomly initialized model.")
    
    sys2_intervener = MetacognitiveIntervention(entropy_threshold=0.5, alpha_sys2=0.85).to(device)
    temp_scaler = TemperatureScaler(init_temp=1.5).to(device)
    
    dummy_prototypes = torch.randn(4, config.get('model', {}).get('hidden_dim', 64)).to(device)

    # 5. Forward Inference
    model.eval()
    with torch.no_grad():
        print("[INFO] Executing forward inference and state monitoring...")
        logits_s, logits_t, emb_final, _ = model(dummy_input, return_emb=True)
        
        logits_s_current = logits_s[:, -1, :] 
        
        calibrated_logits = temp_scaler(logits_s_current)
        sys2_probs, sys2_trigger, entropy = sys2_intervener(calibrated_logits, emb_final, dummy_prototypes)

        final_pred_idx = sys2_probs.argmax(dim=1).item()
        final_stage_name = stage_names_config[final_pred_idx]

    # 6. Evaluation Summary
    print("\n" + "=" * 65)
    print("[SUCCESS] Pipeline executed successfully.")
    print("-" * 65)
    print(f"  > Input Feature Shape: {concept_features.shape}")
    print(f"  > Sys1 Shannon Entropy: {entropy.item():.4f}")
    if sys2_trigger.item() == 1:
        print("  > [ACTION] Entropy threshold exceeded. Sys2 Intervention TRIGGERED.")
    else:
        print("  > [PASS] High cognitive confidence. Sys1 Prediction RETAINED.")
    
    print(f"  > [RESULT] Final Predicted State: {final_stage_name} (Index: {final_pred_idx})")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
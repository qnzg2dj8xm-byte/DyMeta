import os
import yaml
import torch
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from src.utils.common import seed_everything
from src.models.networks import DualMechanismNet
from scripts.train import create_sliding_window_dataset 

def main():
    print("=" * 65)
    print("[INFO] DynaMeta: Evaluation Pipeline")
    print("=" * 65)
    
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_mouse = config['data']['test_mouse_id']
    processed_dir = os.path.join("data/processed", test_mouse)
    
    test_features, test_labels = [], []
    for file_name in os.listdir(processed_dir):
        if file_name.endswith("_features.npy"):
            test_features.append(np.load(os.path.join(processed_dir, file_name)))
            test_labels.append(np.load(os.path.join(processed_dir, file_name.replace("_features.npy", "_labels.npy"))))
            
    if not test_features:
        print("[ERROR] Test set not found!")
        return
        
    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    print(f"[INFO] Test Set Loaded ({test_mouse}): Features {test_features.shape}")

    scaler_path = "checkpoints/feature_scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        test_features = scaler.transform(test_features)
        print("[INFO] Applied fitted RobustScaler to test features.")
    else:
        print("[WARNING] Scaler not found. Proceeding without normalization.")

    seq_len = config['signal']['seq_len_encoder']
    fut_steps = config['model']['future_steps']
    t_X, t_Y_fut, _, _ = create_sliding_window_dataset(test_features, test_labels, seq_len, fut_steps)
    t_X, t_Y_fut = t_X.to(device), t_Y_fut.to(device)

    model = DualMechanismNet(
        input_dim=test_features.shape[1], 
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    weight_path = "checkpoints/dynameta_best.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    print(f"[INFO] Model Weights Loaded from: {weight_path}")

    print("[INFO] Running Full Forward Inference on Test Set...")
    with torch.no_grad():
        logits_s, _ = model(t_X)
        preds = logits_s[:, -1, :].argmax(dim=1).cpu().numpy()
        trues = t_Y_fut.cpu().numpy()

    print("\n" + "=" * 65)
    print(" EVALUATION REPORT")
    print("=" * 65)
    
    acc = accuracy_score(trues, preds)
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")
    
    stage_names = config.get('labels', {}).get('stage_names', ['Microarousal', 'NREM', 'REM', 'Wake'])
    unique_classes = np.unique(np.concatenate((trues, preds)))
    target_names = [stage_names[i] for i in unique_classes] if len(unique_classes) <= len(stage_names) else None
    
    print("Classification Report (Precision / Recall / F1-Score):")
    print("-" * 65)
    print(classification_report(trues, preds, target_names=target_names, zero_division=0))
    print("=" * 65)

if __name__ == "__main__":
    main()
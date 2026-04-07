import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler

from src.utils.common import seed_everything
from src.models.networks import DualMechanismNet, PrototypeRegistry
from src.models.losses import TemporalBehaviorContrastiveLoss, BinaryFocalLoss
from src.engine.trainer import DynaMetaTrainer

def create_sliding_window_dataset(features, labels, seq_len=5, future_steps=5):
    """
    Industrial-grade sliding window generation using memory strides (O(1) complexity).
    Prevents memory explosion and massive CPU bottlenecks on large 24-hour datasets.
    """
    N = len(features)
    valid_len = N - seq_len - future_steps + 1
    
    if valid_len <= 0:
        raise ValueError("Data sequence is too short for the specified window and future steps.")
        
    # 1. Zero-copy Memory View for X_seq
    shape_X = (valid_len, seq_len, features.shape[1])
    strides_X = (features.strides[0], features.strides[0], features.strides[1])
    X_seq = np.lib.stride_tricks.as_strided(features, shape=shape_X, strides=strides_X)
    
    # 2. Vectorized label alignment (Matching exact offsets)
    Y_curr = labels[seq_len - 1 : seq_len - 1 + valid_len]
    Y_fut = labels[seq_len - 1 + future_steps : seq_len - 1 + future_steps + valid_len]
    
    # 3. Concept features matching Y_curr
    C_curr = features[seq_len - 1 : seq_len - 1 + valid_len]
        
    return (torch.tensor(X_seq, dtype=torch.float32), 
            torch.tensor(Y_fut, dtype=torch.long), 
            torch.tensor(Y_curr, dtype=torch.long), 
            torch.tensor(C_curr, dtype=torch.float32))

def main():
    print("=" * 65)
    print("[INFO] DynaMeta: Large-Scale Training Pipeline")
    print("=" * 65)
    
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    seed_everything(config.get('training', {}).get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Compute Device: {device}")

    train_mice = [m for m in config['data']['mouse_ids'] if m != config['data']['test_mouse_id']]
    print(f"[INFO] Training Subject IDs: {train_mice}")
    
    all_features, all_labels = [], []
    for mouse_id in train_mice:
        processed_dir = os.path.join("data/processed", mouse_id)
        if not os.path.exists(processed_dir):
            continue
            
        for file_name in os.listdir(processed_dir):
            if file_name.endswith("_features.npy"):
                feat_path = os.path.join(processed_dir, file_name)
                lbl_path = os.path.join(processed_dir, file_name.replace("_features.npy", "_labels.npy"))
                
                if os.path.exists(lbl_path):
                    all_features.append(np.load(feat_path))
                    all_labels.append(np.load(lbl_path))

    if not all_features:
        print("[ERROR] No training data found. Please run process_data.py first.")
        return

    train_features = np.concatenate(all_features, axis=0)
    train_labels = np.concatenate(all_labels, axis=0)
    print(f"[INFO] Raw Training Set: Features {train_features.shape}, Labels {train_labels.shape}")

    print("[INFO] Applying RobustScaler for feature normalization...")
    scaler = RobustScaler()
    train_features = scaler.fit_transform(train_features)
    
    os.makedirs("checkpoints", exist_ok=True)
    scaler_path = "checkpoints/feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  > Scaler fitted and saved to: {scaler_path}")

    seq_len = config['signal']['seq_len_encoder']
    fut_steps = config['model']['future_steps']
    
    # Lightning-fast memory view execution
    t_X, t_Y_fut, t_Y_curr, t_C = create_sliding_window_dataset(train_features, train_labels, seq_len, fut_steps)
    
    Y_train_is_trans = (t_Y_curr != t_Y_fut).float()
    num_neg = (Y_train_is_trans == 0).sum()
    num_pos = (Y_train_is_trans == 1).sum()
    true_pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)], dtype=torch.float32, device=device)
    print(f"[INFO] Dynamic focal loss pos_weight computed: {true_pos_weight.item():.4f}")

    dataset = TensorDataset(t_X, t_Y_fut, t_Y_curr, t_C)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    model = DualMechanismNet(
        input_dim=train_features.shape[1], 
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    criterion_trans = BinaryFocalLoss(alpha=0.25, gamma=2.0, pos_weight=true_pos_weight).to(device)
    criterion_tbc = TemporalBehaviorContrastiveLoss(temperature=0.07, alpha=0.5).to(device)
    
    proto_registry = PrototypeRegistry(
        num_classes=config['model']['num_classes'], 
        emb_dim=config['model']['hidden_dim'],
        device=device
    )

    trainer = DynaMetaTrainer(model, dataloader, optimizer, criterion_tbc, criterion_trans, proto_registry, device)

    epochs = config.get('training', {}).get('epochs', 15)
    print("\n" + "-" * 65)
    print("[INFO] Starting Large-Scale Training Iterations...")
    print("-" * 65)
    
    for epoch in range(epochs):
        metrics = trainer.train_epoch()
        
        model.eval()
        with torch.no_grad():
            b_X, b_Y_fut, _, _ = next(iter(dataloader))
            b_X, b_Y_fut = b_X.to(device), b_Y_fut.to(device)
            logits_s, _ = model(b_X)
            preds = logits_s[:, -1, :].argmax(dim=1)
            acc = (preds == b_Y_fut).sum().item() / b_Y_fut.size(0) * 100
            
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {metrics['loss']:.4f} (TBC:{metrics['loss_tbc']:.4f}, Trans:{metrics['loss_trans']:.4f}) | Batch Acc: {acc:.1f}%")

    save_path = "checkpoints/dynameta_best.pth"
    torch.save(model.state_dict(), save_path)
    print("-" * 65)
    print(f"[SUCCESS] Training Completed. Weights saved to: {save_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
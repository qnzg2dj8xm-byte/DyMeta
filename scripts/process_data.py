import os
import yaml
import numpy as np
import pandas as pd

from src.data.loader import get_parts_for_mouse, load_data
from src.data.features import extract_concept_features

def main():
    print("[INFO] 多模态数据预处理与特征提取流水线已启动。")
    
    config_path = "configs/default_config.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] 配置文件未找到，请检查路径: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    base_raw_path = "data/raw"
    base_processed_path = "data/processed"
    os.makedirs(base_processed_path, exist_ok=True)
    
    fs_calcium = config.get('signal', {}).get('fs_calcium', 1.92)
    mouse_ids = config.get('data', {}).get('mouse_ids', [])
    
    for mouse_id in mouse_ids:
        print(f"\n[INFO] 当前处理对象 (Subject ID): {mouse_id}")
        parts = get_parts_for_mouse(base_raw_path, mouse_id)
        
        if not parts:
            print(f"[WARNING] 路径 {os.path.join(base_raw_path, mouse_id)} 下未检测到有效分段数据，跳过该对象。")
            continue
            
        for part_path in parts:
            part_name = os.path.basename(part_path)
            print(f"    -> [Loading] 正在解析原始多模态信号矩阵: {part_name}")
            
            try:
                (X, y_sleep, y_beh, beh_cols, stage_names, 
                 raw_eeg, raw_emg, fs_eeg, eeg_indices) = load_data(
                    part_path=part_path, 
                    fs_calcium=fs_calcium
                )
                
                print("    -> [Processing] 正在提取频域功率谱密度 (PSD) 与多模态表征特征...")
                beh_df = pd.DataFrame(y_beh, columns=beh_cols)
                concept_features = extract_concept_features(
                    raw_eeg, eeg_indices, beh_df, fs_eeg=fs_eeg, raw_emg=raw_emg
                )
                
                save_dir = os.path.join(base_processed_path, mouse_id)
                os.makedirs(save_dir, exist_ok=True)
                
                features_save_path = os.path.join(save_dir, f"{part_name}_features.npy")
                labels_save_path = os.path.join(save_dir, f"{part_name}_labels.npy")
                
                np.save(features_save_path, concept_features)
                np.save(labels_save_path, y_sleep)
                
                print(f"    -> [Success] 特征与标签张量已完成序列化，写入至: {save_dir}")
                
            except Exception as e:
                print(f"[ERROR] 处理分段 {part_name} 时发生运行期异常: {str(e)}")

    print("\n[INFO] 数据预处理流水线执行完毕。")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
数据切分脚本：将每个受试者的数据按8:1:1时间段划分
"""
import os
import numpy as np
from pathlib import Path

# 配置
processed_dir = Path("data/processed")
split_dir = Path("data/split_data")
task_name = "auditory"

# 时间段划分（8:1:1）
TRAIN_END = 72952      # 前80%
VAL_END = 82071        # 中间10%
TOTAL_LEN = 91191      # 总长度

# 获取所有MEG文件并按时间戳排序
meg_files = sorted([f for f in os.listdir(processed_dir) if f.endswith("_extracted.npy")])
envelope_file = processed_dir / "envelope_gammatone_64hz.npy"

print(f"找到 {len(meg_files)} 个MEG文件")
print(f"时间段划分：训练[0:{TRAIN_END}] / 验证[{TRAIN_END}:{VAL_END}] / 测试[{VAL_END}:{TOTAL_LEN}]\n")

# 创建目标目录
split_dir.mkdir(exist_ok=True)

# 加载共享的envelope数据
print("加载共享envelope数据...")
envelope_data = np.load(envelope_file)
print(f"Envelope原始形状: {envelope_data.shape}")

# 扩展为 (time, 1) 格式，与模型输出 [B, T, 1] 匹配
envelope_data = envelope_data[:, np.newaxis]  # (91191,) -> (91191, 1)
print(f"Envelope扩展后形状: {envelope_data.shape}\n")

# 切分envelope
envelope_train = envelope_data[:TRAIN_END]
envelope_val = envelope_data[TRAIN_END:VAL_END]
envelope_test = envelope_data[VAL_END:TOTAL_LEN]

print(f"Envelope切分后形状:")
print(f"  训练集: {envelope_train.shape}")
print(f"  验证集: {envelope_val.shape}")
print(f"  测试集: {envelope_test.shape}\n")

# 处理每个受试者
for idx, meg_file in enumerate(meg_files):
    sub_id = idx + 1
    print(f"处理 sub-{sub_id:02d}: {meg_file}")

    # 加载MEG数据
    meg_path = processed_dir / meg_file
    meg_data = np.load(meg_path)
    print(f"  MEG原始形状: {meg_data.shape}")

    # 转置为 (time, channels) 格式
    meg_data = meg_data.T  # (64, 91191) -> (91191, 64)
    print(f"  MEG转置后形状: {meg_data.shape}")

    # 逐通道 z-score 标准化（用训练集的统计量，避免数据泄露）
    train_slice = meg_data[:TRAIN_END, :]
    ch_mean = train_slice.mean(axis=0, keepdims=True)   # (1, 64)
    ch_std  = train_slice.std(axis=0, keepdims=True)
    ch_std[ch_std == 0] = 1.0  # 防止除以零
    meg_data = (meg_data - ch_mean) / ch_std
    print(f"  标准化后 std: {meg_data[:TRAIN_END].std():.4f} (期望≈1)")

    # 切分MEG数据
    meg_train = meg_data[:TRAIN_END, :]
    meg_val = meg_data[TRAIN_END:VAL_END, :]
    meg_test = meg_data[VAL_END:TOTAL_LEN, :]

    # 保存训练集
    train_eeg_path = split_dir / f"train_-_sub-{sub_id:02d}_-_task-{task_name}_-_eeg.npy"
    train_env_path = split_dir / f"train_-_sub-{sub_id:02d}_-_task-{task_name}_-_envelope.npy"
    np.save(train_eeg_path, meg_train)
    np.save(train_env_path, envelope_train)
    print(f"  保存训练集: {meg_train.shape}")

    # 保存验证集
    val_eeg_path = split_dir / f"val_-_sub-{sub_id:02d}_-_task-{task_name}_-_eeg.npy"
    val_env_path = split_dir / f"val_-_sub-{sub_id:02d}_-_task-{task_name}_-_envelope.npy"
    np.save(val_eeg_path, meg_val)
    np.save(val_env_path, envelope_val)
    print(f"  保存验证集: {meg_val.shape}")

    # 保存测试集
    test_eeg_path = split_dir / f"test_-_sub-{sub_id:02d}_-_task-{task_name}_-_eeg.npy"
    test_env_path = split_dir / f"test_-_sub-{sub_id:02d}_-_task-{task_name}_-_envelope.npy"
    np.save(test_eeg_path, meg_test)
    np.save(test_env_path, envelope_test)
    print(f"  保存测试集: {meg_test.shape}\n")

print(f"✓ 完成！共生成 {len(meg_files) * 6} 个文件")
print(f"  - 训练集: {len(meg_files)} 个受试者 × 2 文件 = {len(meg_files) * 2} 个")
print(f"  - 验证集: {len(meg_files)} 个受试者 × 2 文件 = {len(meg_files) * 2} 个")
print(f"  - 测试集: {len(meg_files)} 个受试者 × 2 文件 = {len(meg_files) * 2} 个")

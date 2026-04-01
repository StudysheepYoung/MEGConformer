#!/usr/bin/env python3
"""验证数据格式和形状"""
import numpy as np
from pathlib import Path

split_dir = Path("data/split_data")

# 检查一个训练样本
train_eeg = np.load(split_dir / "train_-_sub-01_-_task-auditory_-_eeg.npy")
train_env = np.load(split_dir / "train_-_sub-01_-_task-auditory_-_envelope.npy")

print("=== 训练集数据形状 ===")
print(f"EEG: {train_eeg.shape} (期望: (72952, 64) - time × channels)")
print(f"Envelope: {train_env.shape} (期望: (72952,) - 1D array)")

# 检查验证集
val_eeg = np.load(split_dir / "val_-_sub-01_-_task-auditory_-_eeg.npy")
val_env = np.load(split_dir / "val_-_sub-01_-_task-auditory_-_envelope.npy")

print("\n=== 验证集数据形状 ===")
print(f"EEG: {val_eeg.shape} (期望: (9119, 64))")
print(f"Envelope: {val_env.shape} (期望: (9119,))")

# 检查测试集
test_eeg = np.load(split_dir / "test_-_sub-01_-_task-auditory_-_eeg.npy")
test_env = np.load(split_dir / "test_-_sub-01_-_task-auditory_-_envelope.npy")

print("\n=== 测试集数据形状 ===")
print(f"EEG: {test_eeg.shape} (期望: (9120, 64))")
print(f"Envelope: {test_env.shape} (期望: (9120,))")

# 验证数据连续性（检查是否正确切分）
print("\n=== 验证数据连续性 ===")
total_time = train_eeg.shape[0] + val_eeg.shape[0] + test_eeg.shape[0]
print(f"总时间步: {total_time} (期望: 91191)")
print(f"训练集占比: {train_eeg.shape[0]/total_time*100:.1f}% (期望: 80.0%)")
print(f"验证集占比: {val_eeg.shape[0]/total_time*100:.1f}% (期望: 10.0%)")
print(f"测试集占比: {test_eeg.shape[0]/total_time*100:.1f}% (期望: 10.0%)")

# 检查数据范围
print("\n=== 数据范围检查 ===")
print(f"训练集 EEG: [{train_eeg.min():.3f}, {train_eeg.max():.3f}]")
print(f"训练集 Envelope: [{train_env.min():.3f}, {train_env.max():.3f}]")

print("\n✓ 数据格式验证完成！")

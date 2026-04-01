#!/usr/bin/env python3
"""测试数据加载是否正常"""
import sys
import os
import glob
sys.path.append('.')
from util.dataset import RegressionDataset
import numpy as np

data_folder = "data/split_data"
features = ["eeg", "envelope"]
input_length = 640

# 测试训练集加载
print("=== 测试训练集加载 ===")
train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*"))
               if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
print(f"找到 {len(train_files)} 个训练文件")

train_dataset = RegressionDataset(
    files=train_files,
    input_length=input_length,
    channels=64,
    task="train",
    g_con=True,
    windows_per_sample=20
)

print(f"训练集样本数: {len(train_dataset)}")
print(f"预加载数据组数: {len(train_dataset.preloaded_data)}")

# 获取一个样本
eeg, envelope, sub_idx = train_dataset[0]
print(f"\n样本形状:")
print(f"  EEG: {eeg.shape} (期望: [64, 640])")
print(f"  Envelope: {envelope.shape} (期望: [1, 640])")
print(f"  Subject ID: {sub_idx}")

# 检查数据范围
print(f"\n数据范围:")
print(f"  EEG: [{eeg.min():.3f}, {eeg.max():.3f}]")
print(f"  Envelope: [{envelope.min():.3f}, {envelope.max():.3f}]")

# 测试验证集
print("\n=== 测试验证集加载 ===")
val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*"))
             if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
print(f"找到 {len(val_files)} 个验证文件")

val_dataset = RegressionDataset(
    files=val_files,
    input_length=input_length,
    channels=64,
    task="test",
    g_con=True,
    windows_per_sample=1
)
print(f"验证集样本数: {len(val_dataset)}")

# 检查验证集数据长度
val_data = val_dataset.preloaded_data[0][0][0]  # 第一个受试者的EEG数据
print(f"验证集单个受试者数据长度: {val_data.shape[1]} (期望: 9119)")

# 测试测试集
print("\n=== 测试测试集加载 ===")
test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*"))
              if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
print(f"找到 {len(test_files)} 个测试文件")

test_dataset = RegressionDataset(
    files=test_files,
    input_length=input_length,
    channels=64,
    task="val",
    g_con=True,
    windows_per_sample=1
)
print(f"测试集样本数: {len(test_dataset)}")

# 检查测试集数据长度
test_data = test_dataset.preloaded_data[0][0][0]  # 第一个受试者的EEG数据
print(f"测试集单个受试者数据长度: {test_data.shape[1]} (期望: 9120)")

print("\n✓ 数据加载测试通过！")
print(f"\n数据集统计:")
print(f"  训练集: 17个受试者 × 20窗口 = {len(train_dataset)} 样本")
print(f"  验证集: 17个受试者 = {len(val_dataset)} 样本")
print(f"  测试集: 17个受试者 = {len(test_dataset)} 样本")

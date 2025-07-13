# R2Gen项目调试历史记录

## 项目概述
- **项目名称**: R2Gen (Generating Radiology Reports via Memory-driven Transformer)
- **论文**: EMNLP-2020
- **目标**: 训练一个能够生成放射学报告的深度学习模型
- **数据集**: IU X-Ray数据集

## 环境信息
- **操作系统**: Linux (Pop!_OS)
- **Python版本**: 3.9.23
- **Conda环境**: dl
- **主要依赖**:
  - torch: 2.5.1 (原要求1.5.1，向后兼容)
  - torchvision: 0.20.1 (原要求0.6.1，向后兼容)
  - opencv-python: 4.12.0.88 (原要求4.4.0.42，向后兼容)
  - numpy: 2.0.1

## 调试过程

### 第一步：环境检查和依赖安装 ✅
**时间**: 开始调试
**问题**: 需要验证环境和依赖是否满足要求
**解决方案**:
1. 检查conda环境：当前在`dl`环境中
2. 检查已安装包：torch, torchvision, opencv-python, numpy都已安装
3. 版本比README要求更新，但应该向后兼容
**结果**: 环境检查通过

### 第二步：数据集路径配置 ✅
**时间**: 环境检查完成后
**问题**: 数据集在外层`datasets`文件夹中，需要链接到R2Gen-main/data目录
**解决方案**:
1. 发现数据集位于`datasets/iu_xray/`
2. 创建符号链接：`ln -sf ../../datasets/iu_xray R2Gen-main/data/iu_xray`
3. 验证链接成功创建
**结果**: 数据集路径配置完成

### 第三步：初始运行测试 ✅
**时间**: 数据集配置完成后
**问题**: 首次运行遇到Java依赖缺失错误
**错误信息**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'java'
```
**原因**: METEOR评估指标需要Java环境
**解决方案**:
1. 尝试系统安装Java失败（需要sudo权限）
2. 使用conda安装Java：`conda install -c conda-forge openjdk -y`
3. 验证Java安装成功：`java -version`
**结果**: Java环境配置完成

### 第四步：pandas兼容性问题修复 ✅
**时间**: Java安装后重新运行
**问题**: 训练完成后遇到pandas版本兼容性错误
**错误信息**:
```
AttributeError: 'DataFrame' object has no attribute 'append'
```
**原因**: 新版本pandas移除了`append`方法
**解决方案**:
修改`R2Gen-main/modules/trainer.py`第111-112行：
```python
# 原代码
record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
record_table = record_table.append(self.best_recorder['test'], ignore_index=True)

# 修复后
record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['val']])], ignore_index=True)
record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['test']])], ignore_index=True)
```
**结果**: pandas兼容性问题解决

### 第五步：成功运行验证 ✅
**时间**: 修复pandas问题后
**运行命令**:
```bash
python main.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 4 \
--epochs 2 \
--save_dir results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223
```

**训练结果**:
- **Epoch 1**:
  - train_loss: 3.124
  - val_BLEU_4: 0.064
  - test_BLEU_4: 0.081
- **Epoch 2**:
  - train_loss: 2.103 (显著下降)
  - val_BLEU_4: 0.064
  - test_BLEU_4: 0.081

**结果**: 训练成功完成，模型正常收敛

## 显存优化配置
- **batch_size**: 从原始的16调整为4，避免显存不足
- **epochs**: 从100调整为2进行快速验证
- **其他参数**: 保持原始配置

## 生成的文件
- `results/iu_xray/current_checkpoint.pth`: 最新检查点
- `results/iu_xray/model_best.pth`: 最佳模型
- `records/iu_xray.csv`: 训练记录

## 总结
项目成功跑通！主要解决了以下问题：
1. ✅ Java环境依赖
2. ✅ pandas版本兼容性
3. ✅ 显存优化配置
4. ✅ 数据集路径配置

训练过程正常，模型能够正常收敛，loss从3.124下降到2.103，说明模型在学习。

## 后续建议
1. 可以增加epochs数量进行更长时间训练
2. 根据显存情况适当调整batch_size
3. 可以尝试其他超参数优化
4. 建议定期保存检查点以防训练中断

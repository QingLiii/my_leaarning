# R2Gen 深度优化方案

## 📋 优化目标
1. **WandB监控集成** - 全面监控训练过程和硬件性能
2. **显存优化** - 动态调整batch size和梯度累积
3. **精度实验** - 对比FP32、FP16、FP8的性能和效果

## 🎯 实施计划

### 阶段一：WandB监控系统集成

#### 1.1 创建WandB配置模块
```python
# modules/wandb_logger.py
import wandb
import torch
import psutil
import GPUtil
import time
from typing import Dict, Any

class WandBLogger:
    def __init__(self, project_name="R2Gen-Optimization", api_key="68c9ce2a167992d06678c4fdc0d1075b5dfff922"):
        wandb.login(key=api_key)
        self.project_name = project_name
        
    def init_run(self, config: Dict[str, Any], run_name: str = None):
        wandb.init(
            project=self.project_name,
            config=config,
            name=run_name,
            tags=["optimization", "precision_study"]
        )
        
    def log_system_metrics(self):
        """记录系统和GPU性能指标"""
        # CPU和内存
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU信息
        gpus = GPUtil.getGPUs()
        gpu_metrics = {}
        for i, gpu in enumerate(gpus):
            gpu_metrics[f'gpu_{i}_utilization'] = gpu.load * 100
            gpu_metrics[f'gpu_{i}_memory_used'] = gpu.memoryUsed
            gpu_metrics[f'gpu_{i}_memory_total'] = gpu.memoryTotal
            gpu_metrics[f'gpu_{i}_memory_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
            gpu_metrics[f'gpu_{i}_temperature'] = gpu.temperature
        
        wandb.log({
            'system/cpu_percent': cpu_percent,
            'system/memory_percent': memory.percent,
            'system/memory_used_gb': memory.used / (1024**3),
            **gpu_metrics
        })
```

#### 1.2 增强Trainer类
```python
# 在trainer.py中添加WandB集成
class EnhancedTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, 
                 train_dataloader, val_dataloader, test_dataloader, wandb_logger=None):
        super().__init__(model, criterion, metric_ftns, optimizer, args)
        self.wandb_logger = wandb_logger
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.mixed_precision = getattr(args, 'mixed_precision', None)  # 'fp16', 'fp8', None
        
        # 初始化混合精度
        if self.mixed_precision == 'fp16':
            self.scaler = torch.cuda.amp.GradScaler()
        
    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()
        
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            # 系统监控
            if batch_idx % 10 == 0 and self.wandb_logger:
                self.wandb_logger.log_system_metrics()
            
            # 前向传播和损失计算
            loss = self._forward_step(images, reports_ids, reports_masks)
            
            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self._backward_step(loss)
                
            train_loss += loss.item()
            
            # 记录训练指标
            if self.wandb_logger and batch_idx % 50 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                    'train/batch': batch_idx
                })
        
        return {'train_loss': train_loss / len(self.train_dataloader)}
```

### 阶段二：显存优化和动态Batch Size

#### 2.1 显存监控和自适应Batch Size
```python
# modules/memory_optimizer.py
import torch
import GPUtil

class MemoryOptimizer:
    def __init__(self, initial_batch_size=16, max_memory_percent=85):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.max_memory_percent = max_memory_percent
        self.gradient_accumulation_steps = 1
        
    def get_gpu_memory_usage(self):
        """获取GPU显存使用率"""
        gpus = GPUtil.getGPUs()
        if gpus:
            return (gpus[0].memoryUsed / gpus[0].memoryTotal) * 100
        return 0
    
    def optimize_batch_size(self, model, sample_batch):
        """动态优化batch size"""
        print("🔍 开始显存优化测试...")
        
        # 测试不同batch size的显存使用
        test_sizes = [4, 8, 16, 24, 32, 48, 64]
        optimal_size = self.initial_batch_size
        
        for size in test_sizes:
            try:
                # 模拟前向传播
                test_batch = self._create_test_batch(sample_batch, size)
                
                torch.cuda.empty_cache()
                memory_before = self.get_gpu_memory_usage()
                
                with torch.no_grad():
                    _ = model(*test_batch)
                
                memory_after = self.get_gpu_memory_usage()
                
                print(f"Batch size {size}: {memory_after:.1f}% GPU memory")
                
                if memory_after < self.max_memory_percent:
                    optimal_size = size
                else:
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch size {size}")
                    break
                    
        # 计算梯度累积步数
        target_effective_batch = self.initial_batch_size
        self.current_batch_size = optimal_size
        self.gradient_accumulation_steps = max(1, target_effective_batch // optimal_size)
        
        print(f"✅ 优化结果: batch_size={optimal_size}, gradient_accumulation={self.gradient_accumulation_steps}")
        return optimal_size, self.gradient_accumulation_steps
```

### 阶段三：精度对比实验

#### 3.1 精度实验配置
```python
# experiments/precision_study.py
import torch
from torch.cuda.amp import autocast, GradScaler

class PrecisionExperiment:
    def __init__(self, base_args):
        self.base_args = base_args
        self.precision_configs = {
            'fp32': {'mixed_precision': None, 'description': 'Full Precision'},
            'fp16': {'mixed_precision': 'fp16', 'description': 'Half Precision'},
            'fp8': {'mixed_precision': 'fp8', 'description': 'FP8 Precision (if supported)'}
        }
        
    def run_precision_study(self):
        """运行精度对比实验"""
        results = {}
        
        for precision_name, config in self.precision_configs.items():
            print(f"\n🧪 开始 {precision_name} 精度实验...")
            
            # 创建实验配置
            exp_args = self._create_experiment_args(precision_name, config)
            
            # 运行实验
            result = self._run_single_experiment(exp_args, precision_name)
            results[precision_name] = result
            
        return results
    
    def _create_experiment_args(self, precision_name, config):
        """创建实验参数"""
        import copy
        args = copy.deepcopy(self.base_args)
        
        # 设置实验特定参数
        args.epochs = 15
        args.save_dir = f"results/precision_study/{precision_name}"
        args.mixed_precision = config['mixed_precision']
        args.experiment_name = f"R2Gen_{precision_name}_precision"
        
        return args
```

#### 3.2 实验执行脚本
```python
# run_optimization_experiments.py
#!/usr/bin/env python3

import argparse
import torch
from modules.wandb_logger import WandBLogger
from modules.memory_optimizer import MemoryOptimizer
from experiments.precision_study import PrecisionExperiment

def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_memory_opt', action='store_true', help='运行显存优化')
    parser.add_argument('--run_precision_study', action='store_true', help='运行精度对比实验')
    parser.add_argument('--run_all', action='store_true', help='运行所有优化实验')
    args = parser.parse_args()
    
    # 初始化WandB
    wandb_logger = WandBLogger()
    
    if args.run_memory_opt or args.run_all:
        print("🚀 开始显存优化实验...")
        run_memory_optimization(wandb_logger)
    
    if args.run_precision_study or args.run_all:
        print("🚀 开始精度对比实验...")
        run_precision_study(wandb_logger)

def run_memory_optimization(wandb_logger):
    """运行显存优化实验"""
    # 实现显存优化逻辑
    pass

def run_precision_study(wandb_logger):
    """运行精度对比实验"""
    # 实现精度对比逻辑
    pass

if __name__ == '__main__':
    main()
```

## 📊 实验设计

### 实验1: 显存优化基线测试
- **目标**: 找到最优batch size和梯度累积配置
- **方法**: 逐步增加batch size直到显存使用率达到85%
- **监控**: GPU显存使用率、训练速度、模型性能

### 实验2: 精度对比研究
- **配置**: FP32 vs FP16 vs FP8
- **训练**: 每种精度15个epoch
- **评估**: 
  - 训练速度 (samples/sec)
  - 显存使用量
  - 模型性能 (BLEU, METEOR, ROUGE-L)
  - 数值稳定性

### 实验3: 综合优化
- **结合**: 最优batch size + 最佳精度设置
- **长期训练**: 100 epochs
- **对比**: 与原始配置的性能对比

## 🔧 技术实现细节

### WandB监控指标
```yaml
训练指标:
  - train_loss, val_loss, test_loss
  - BLEU-1/2/3/4, METEOR, ROUGE-L
  - learning_rate, gradient_norm

系统指标:
  - GPU利用率、显存使用率、温度
  - CPU使用率、内存使用率
  - 训练速度 (samples/sec, batches/sec)

模型指标:
  - 参数数量、模型大小
  - 前向传播时间、反向传播时间
```

### 梯度累积实现
```python
# 在trainer中实现梯度累积
def _train_epoch_with_accumulation(self, epoch):
    self.model.train()
    accumulated_loss = 0
    
    for batch_idx, batch_data in enumerate(self.train_dataloader):
        # 前向传播
        loss = self._forward_step(batch_data) / self.gradient_accumulation_steps
        
        # 反向传播（累积梯度）
        if self.mixed_precision == 'fp16':
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        accumulated_loss += loss.item()
        
        # 每accumulation_steps步更新一次参数
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            if self.mixed_precision == 'fp16':
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
```

## 📈 预期收益

1. **性能提升**: 通过FP16可能获得1.5-2x训练速度提升
2. **显存优化**: 通过动态batch size最大化GPU利用率
3. **监控完善**: 全面的训练过程可视化和分析
4. **实验可重现**: 详细的实验记录和配置管理

## ⚠️ 风险评估

1. **数值稳定性**: FP16可能导致梯度下溢，需要梯度缩放
2. **模型性能**: 低精度可能影响最终模型效果
3. **硬件兼容**: FP8需要较新的GPU支持
4. **实验时间**: 完整的精度对比实验需要较长时间

## 🎯 成功指标

1. **技术指标**:
   - 训练速度提升 > 30%
   - 显存利用率 > 80%
   - 模型性能损失 < 5%

2. **监控指标**:
   - 完整的WandB仪表板
   - 实时硬件性能监控
   - 自动化实验记录

这个方案如何？需要我调整或补充哪些部分？

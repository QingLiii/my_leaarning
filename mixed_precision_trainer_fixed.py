#!/usr/bin/env python3
"""
支持混合精度的R2Gen训练器
修复BatchNorm兼容性问题
"""

import sys
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import os
from datetime import datetime

sys.path.append('R2Gen-main')

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.loss import compute_loss
from models.r2gen import R2GenModel
from modules.wandb_logger import WandBLogger
from modules.enhanced_trainer import EnhancedTrainer

class MixedPrecisionConfig:
    """混合精度训练配置"""
    
    def __init__(self, precision='fp32', epochs=15):
        self.precision = precision
        self.epochs = epochs
        
        # 数据集配置
        self.image_dir = 'datasets/iu_xray/images/'
        self.ann_path = 'datasets/iu_xray/annotation.json'
        self.dataset_name = 'iu_xray'
        self.max_seq_length = 60
        self.threshold = 3
        self.num_workers = 2
        
        # 根据精度设置batch size
        if precision == 'fp32':
            self.batch_size = 12
        elif precision == 'fp16':
            self.batch_size = 16
        else:  # fp8
            self.batch_size = 20
        
        # 训练配置
        self.seed = 9223
        self.n_gpu = 1
        self.save_period = 1
        self.monitor_mode = 'max'
        self.monitor_metric = 'BLEU_4'
        self.early_stop = 50
        self.resume = None
        self.validate_every = 1
        
        # 优化器配置 - 严格按论文
        self.optim = 'Adam'
        self.lr_ve = 5e-5
        self.lr_ed = 1e-4
        self.weight_decay = 0
        self.amsgrad = True
        
        # 学习率调度 - 每epoch衰减0.8
        self.lr_scheduler = 'StepLR'
        self.step_size = 1
        self.gamma = 0.8
        
        # 模型配置
        self.d_model = 512
        self.d_ff = 512
        self.d_vf = 2048
        self.num_heads = 8
        self.num_layers = 3
        self.dropout = 0.1
        self.logit_layers = 1
        self.bos_idx = 0
        self.eos_idx = 0
        self.pad_idx = 0
        self.use_bn = True
        self.drop_prob_lm = 0.5
        
        # 记忆模块
        self.rm_num_slots = 3
        self.rm_num_heads = 8
        self.rm_d_model = 512
        
        # 视觉特征提取
        self.visual_extractor = 'resnet101'
        self.visual_extractor_pretrained = True
        
        # 采样配置
        self.sample_method = 'beam_search'
        self.beam_size = 3
        self.temperature = 1.0
        self.sample_n = 1
        
        # 混合精度配置
        if precision == 'fp16':
            self.mixed_precision = True
            self.precision_mode = 'fp16'
        elif precision == 'fp8':
            self.mixed_precision = True
            self.precision_mode = 'fp8'
        else:
            self.mixed_precision = False
            self.precision_mode = 'fp32'
        
        # 保存路径
        self.save_dir = f'results/mixed_precision_fixed/{precision}'
        self.record_dir = f'results/mixed_precision_fixed/{precision}'
        self.experiment_name = f'R2Gen_{precision}_fixed'

def run_mixed_precision_experiment(precision='fp32', epochs=15):
    """运行混合精度实验"""
    
    print(f"\n{'='*60}")
    print(f"🚀 开始 {precision.upper()} 混合精度实验 (修复版)")
    print(f"{'='*60}")
    
    # 创建配置
    config = MixedPrecisionConfig(precision, epochs)
    
    print(f"📋 实验配置:")
    print(f"   精度: {precision}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   混合精度: {config.mixed_precision}")
    
    try:
        # 创建数据加载器
        print(f"📊 创建数据加载器...")
        tokenizer = Tokenizer(config)
        train_dataloader = R2DataLoader(config, tokenizer, split='train', shuffle=True)
        val_dataloader = R2DataLoader(config, tokenizer, split='val', shuffle=False)
        test_dataloader = R2DataLoader(config, tokenizer, split='test', shuffle=False)
        
        # 创建模型
        print(f"🏗️ 创建模型...")
        model = R2GenModel(config, tokenizer).cuda()
        
        # 创建优化器
        optimizer = build_optimizer(config, model)
        lr_scheduler = build_lr_scheduler(config, optimizer)
        
        # 初始化WandB
        print(f"📈 初始化WandB...")
        os.environ['WANDB_API_KEY'] = '68c9ce2a167992d06678c4fdc0d1075b5dfff922'
        wandb_logger = WandBLogger(project_name="R2Gen-Mixed-Precision-Fixed")
        
        wandb_config = {
            'precision': precision,
            'mixed_precision': config.mixed_precision,
            'batch_size': config.batch_size,
            'epochs': epochs,
            'learning_rate_ve': config.lr_ve,
            'learning_rate_ed': config.lr_ed,
            'lr_scheduler': config.lr_scheduler,
            'lr_decay_factor': config.gamma,
            'model': 'R2Gen_Fixed',
            'dataset': config.dataset_name,
            'experiment_type': 'mixed_precision_fixed'
        }
        
        run_name = f"R2Gen_{precision}_fixed_bs{config.batch_size}_ep{epochs}"
        wandb_logger.init_run(wandb_config, run_name=run_name)
        
        # 创建训练器
        print(f"🎯 创建训练器...")
        trainer = EnhancedTrainer(
            model=model,
            criterion=compute_loss,
            metric_ftns=compute_scores,
            optimizer=optimizer,
            args=config,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            wandb_logger=wandb_logger,
            enable_wandb=True
        )
        
        # 开始训练
        print(f"🏃 开始训练...")
        start_time = time.time()
        trainer.train()
        training_time = (time.time() - start_time) / 3600
        
        print(f"✅ {precision.upper()} 训练成功完成!")
        print(f"⏱️ 训练时间: {training_time:.2f} 小时")
        
        wandb_logger.finish()
        
        return {
            'precision': precision,
            'status': 'success',
            'training_time_hours': training_time,
            'batch_size': config.batch_size,
            'epochs': epochs,
            'model_path': f'{config.save_dir}/model_best.pth'
        }
        
    except Exception as e:
        print(f"❌ {precision.upper()} 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'precision': precision,
            'status': 'failed',
            'error': str(e),
            'batch_size': config.batch_size,
            'epochs': epochs
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='混合精度训练实验')
    parser.add_argument('--precision', type=str, default='fp32', 
                       choices=['fp32', 'fp16', 'fp8'], help='训练精度')
    parser.add_argument('--epochs', type=int, default=15, help='训练epoch数')
    
    args = parser.parse_args()
    
    result = run_mixed_precision_experiment(args.precision, args.epochs)
    print(f"\n📊 实验结果: {result}")

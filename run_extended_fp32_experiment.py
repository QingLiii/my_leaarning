#!/usr/bin/env python3
"""
扩展的FP32实验 - 基于成功的5epoch配置运行15epoch
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

# 添加项目路径
sys.path.append('R2Gen-main')

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.loss import compute_loss
from models.r2gen import R2GenModel
from modules.wandb_logger import WandBLogger
from modules.memory_monitor import MemoryMonitor
from modules.enhanced_trainer import EnhancedTrainer

class ExtendedExperimentConfig:
    """扩展实验配置 - 基于成功的配置"""
    
    def __init__(self, epochs=15):
        # 基础配置
        self.epochs = epochs
        
        # 数据集相关
        self.image_dir = 'datasets/iu_xray/images/'
        self.ann_path = 'datasets/iu_xray/annotation.json'
        self.dataset_name = 'iu_xray'
        self.max_seq_length = 60
        self.threshold = 3
        self.num_workers = 2
        self.batch_size = 12
        
        # 训练相关
        self.seed = 9223
        self.n_gpu = 1
        self.save_period = 1
        self.monitor_mode = 'max'
        self.monitor_metric = 'BLEU_4'
        self.early_stop = 50
        self.resume = None
        self.validate_every = 1  # 每个epoch验证
        
        # 优化器相关 - 按论文配置
        self.optim = 'Adam'
        self.lr_ve = 5e-5      # visual extractor学习率
        self.lr_ed = 1e-4      # encoder-decoder学习率
        self.weight_decay = 0
        self.amsgrad = True
        
        # 学习率调度 - 按论文要求每epoch衰减0.8
        self.lr_scheduler = 'StepLR'
        self.step_size = 1     # 每个epoch衰减
        self.gamma = 0.8       # 衰减因子0.8
        
        # 模型相关
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
        
        # 采样相关
        self.sample_method = 'beam_search'
        self.beam_size = 3
        self.temperature = 1.0
        self.sample_n = 1
        
        # 混合精度配置
        self.mixed_precision = False
        self.precision_mode = 'fp32'
        
        # 保存路径
        self.save_dir = f'results/extended_experiment/fp32'
        self.record_dir = f'results/extended_experiment/fp32'
        self.experiment_name = f'R2Gen_fp32_extended'

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='扩展FP32实验')
    parser.add_argument('--epochs', type=int, default=15, help='训练epoch数')
    
    args = parser.parse_args()
    
    print("🚀 R2Gen扩展FP32实验")
    print(f"   训练epochs: {args.epochs}")
    print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 验证前置条件
    print(f"\n📋 验证前置条件...")
    
    # 验证tokenizer解码
    test_args = ExtendedExperimentConfig()
    tokenizer = Tokenizer(test_args)
    test_text = 'the lungs are clear'
    encoded = tokenizer(test_text)
    decoded = tokenizer.decode(encoded)
    
    if len(decoded) == 0:
        print("❌ Tokenizer解码失败，实验终止")
        return
    else:
        print("✅ Tokenizer解码正常")
    
    # 创建配置
    config = ExtendedExperimentConfig(args.epochs)
    
    print(f"\n🚀 开始FP32扩展训练...")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   学习率调度: 每epoch衰减{config.gamma}")
    
    # 创建数据加载器
    print(f"   📊 创建数据加载器...")
    train_dataloader = R2DataLoader(config, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(config, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(config, tokenizer, split='test', shuffle=False)
    
    print(f"   训练样本: {len(train_dataloader.dataset)}")
    print(f"   验证样本: {len(val_dataloader.dataset)}")
    print(f"   测试样本: {len(test_dataloader.dataset)}")
    
    # 创建模型
    print(f"   🏗️ 创建模型...")
    model = R2GenModel(config, tokenizer).cuda()
    
    # 创建优化器
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_lr_scheduler(config, optimizer)
    
    # 初始化WandB
    print(f"   📈 初始化WandB监控...")
    import os
    os.environ['WANDB_API_KEY'] = '68c9ce2a167992d06678c4fdc0d1075b5dfff922'
    wandb_logger = WandBLogger(project_name="R2Gen-Extended-Experiment")
    
    wandb_config = {
        'precision': 'fp32',
        'mixed_precision': False,
        'batch_size': config.batch_size,
        'epochs': args.epochs,
        'learning_rate_ve': config.lr_ve,
        'learning_rate_ed': config.lr_ed,
        'lr_scheduler': config.lr_scheduler,
        'lr_decay_factor': config.gamma,
        'lr_decay_step': config.step_size,
        'model': 'R2Gen',
        'dataset': config.dataset_name,
        'experiment_type': 'extended_fp32',
        'optimizer': config.optim,
        'rm_num_slots': config.rm_num_slots,
        'rm_d_model': config.rm_d_model,
        'beam_size': config.beam_size,
        'd_model': config.d_model,
        'num_layers': config.num_layers,
        'num_heads': config.num_heads
    }
    
    run_name = f"R2Gen_fp32_extended_bs{config.batch_size}_ep{args.epochs}"
    wandb_logger.init_run(wandb_config, run_name=run_name)
    
    # 创建训练器
    print(f"   🎯 创建训练器...")
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
    print(f"   🏃 开始训练...")
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = (time.time() - start_time) / 3600  # 转换为小时
        
        print(f"   ✅ 扩展FP32训练成功完成!")
        wandb_logger.finish()
        
        # 保存结果
        result = {
            'precision': 'fp32',
            'status': 'success',
            'training_time_hours': training_time,
            'batch_size': config.batch_size,
            'epochs': args.epochs,
            'model_path': f'{config.save_dir}/model_best.pth',
            'lr_decay_factor': config.gamma,
            'lr_decay_step': config.step_size
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'extended_fp32_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n🎉 扩展FP32实验完成!")
        print(f"   ⏱️ 训练时间: {training_time:.2f} 小时")
        print(f"   📁 模型保存: {result['model_path']}")
        print(f"   📊 结果文件: {results_file}")
        
    except Exception as e:
        print(f"   ❌ 扩展FP32训练失败: {str(e)}")
        wandb_logger.finish()
        
        # 保存错误信息
        result = {
            'precision': 'fp32',
            'status': 'failed',
            'error': str(e),
            'batch_size': config.batch_size,
            'epochs': args.epochs
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'extended_fp32_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"   📊 错误记录: {results_file}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
完整的精度对比实验脚本
运行FP32 vs FP16 vs FP8的15 epoch训练对比，包含报告质量评估
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime
import time

# 添加R2Gen路径
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


def create_args(precision='fp32', batch_size=12, epochs=15):
    """创建训练参数"""
    class Args:
        def __init__(self):
            # 数据相关
            self.image_dir = 'R2Gen-main/data/iu_xray/images/'
            self.ann_path = 'R2Gen-main/data/iu_xray/annotation.json'
            self.dataset_name = 'iu_xray'
            self.max_seq_length = 60
            self.threshold = 3
            self.num_workers = 2
            self.batch_size = batch_size
            
            # 训练相关
            self.epochs = epochs
            self.seed = 9223
            self.n_gpu = 1
            self.save_period = epochs  # 只在最后保存
            self.monitor_mode = 'max'
            self.monitor_metric = 'BLEU_4'
            self.early_stop = epochs + 10  # 禁用早停
            self.resume = None
            self.validate_every = epochs  # 只在最后验证
            
            # 优化器相关
            self.optim = 'Adam'
            self.lr_ve = 5e-5
            self.lr_ed = 1e-4
            self.weight_decay = 0
            self.amsgrad = True
            
            # 学习率调度
            self.lr_scheduler = 'StepLR'
            self.step_size = 50
            self.gamma = 0.1
            
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
            self.use_bn = 0
            self.drop_prob_lm = 0.5
            
            # 视觉特征提取
            self.visual_extractor = 'resnet101'
            self.visual_extractor_pretrained = True
            
            # 记忆模块
            self.rm_num_slots = 3
            self.rm_num_heads = 8
            self.rm_d_model = 512
            
            # 采样相关
            self.sample_method = 'beam_search'
            self.beam_size = 3
            self.temperature = 1.0
            self.sample_n = 1
            self.group_size = 1
            self.output_logsoftmax = 1
            self.decoding_constraint = 0
            self.block_trigrams = 1
            
            # 精度相关
            self.mixed_precision = None if precision == 'fp32' else precision
            
            # 保存路径
            self.save_dir = f'results/precision_comparison/{precision}'
            self.record_dir = f'results/precision_comparison/{precision}'
            self.experiment_name = f'R2Gen_{precision}_precision'
            
            # 优化相关
            self.gradient_accumulation_steps = 1
            self.log_interval = 50
    
    return Args()


def run_single_precision_training(precision, batch_size, epochs=15):
    """运行单个精度的训练"""
    print(f"\n🚀 开始 {precision.upper()} 精度训练...")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建参数
    args = create_args(precision, batch_size, epochs)
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建数据加载器
    print(f"   📊 创建数据加载器...")
    tokenizer = Tokenizer(args)
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    
    print(f"   训练样本: {len(train_dataloader.dataset)}")
    print(f"   验证样本: {len(val_dataloader.dataset)}")
    print(f"   测试样本: {len(test_dataloader.dataset)}")
    
    # 创建模型
    print(f"   🏗️ 创建模型...")
    model = R2GenModel(args, tokenizer).cuda()
    
    # 创建优化器
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    
    # 初始化WandB
    print(f"   📈 初始化WandB监控...")
    wandb_logger = WandBLogger(project_name="R2Gen-Precision-Comparison")
    
    wandb_config = {
        'precision': precision,
        'mixed_precision': args.mixed_precision,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate_ve': args.lr_ve,
        'learning_rate_ed': args.lr_ed,
        'model': 'R2Gen',
        'dataset': args.dataset_name,
        'experiment_type': 'precision_comparison',
        'optimizer': args.optim,
        'scheduler': args.lr_scheduler
    }
    
    run_name = f"R2Gen_{precision}_bs{batch_size}_ep{epochs}"
    wandb_logger.init_run(wandb_config, run_name=run_name)
    
    # 创建增强版trainer
    print(f"   🎯 创建训练器...")
    trainer = EnhancedTrainer(
        model=model,
        criterion=compute_loss,
        metric_ftns=compute_scores,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        wandb_logger=wandb_logger,
        enable_wandb=True
    )
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 开始训练
        print(f"   🏃 开始训练...")
        trainer.train()
        
        training_success = True
        print(f"   ✅ {precision.upper()} 训练成功完成!")
        
    except Exception as e:
        print(f"   ❌ {precision.upper()} 训练失败: {e}")
        training_success = False
        import traceback
        traceback.print_exc()
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    
    # 结束WandB运行
    wandb_logger.finish()
    
    # 返回结果
    result = {
        'precision': precision,
        'batch_size': batch_size,
        'epochs': epochs,
        'success': training_success,
        'total_time_seconds': total_time,
        'total_time_hours': total_time / 3600,
        'start_time': datetime.fromtimestamp(start_time).isoformat(),
        'end_time': datetime.fromtimestamp(end_time).isoformat(),
        'model_path': os.path.join(args.save_dir, 'model_best.pth') if training_success else None,
        'wandb_run_name': run_name
    }
    
    print(f"   ⏱️ 训练耗时: {total_time/3600:.2f} 小时")
    
    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='R2Gen精度对比实验')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'fp8', 'all'], 
                       default='all', help='运行特定精度实验')
    parser.add_argument('--epochs', type=int, default=15, help='训练epochs数')
    parser.add_argument('--fp32-batch-size', type=int, default=12, help='FP32 batch size')
    parser.add_argument('--fp16-batch-size', type=int, default=24, help='FP16 batch size')
    parser.add_argument('--fp8-batch-size', type=int, default=32, help='FP8 batch size')
    
    args = parser.parse_args()
    
    print(f"🚀 R2Gen精度对比实验")
    print(f"   精度范围: {args.precision}")
    print(f"   训练epochs: {args.epochs}")
    print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 精度配置
    precision_configs = {
        'fp32': {'batch_size': args.fp32_batch_size, 'description': 'Full Precision (FP32)'},
        'fp16': {'batch_size': args.fp16_batch_size, 'description': 'Half Precision (FP16)'},
        'fp8': {'batch_size': args.fp8_batch_size, 'description': 'FP8 Precision (experimental)'}
    }
    
    # 确定要运行的精度
    if args.precision == 'all':
        precisions_to_run = ['fp32', 'fp16', 'fp8']
    else:
        precisions_to_run = [args.precision]
    
    print(f"   将运行精度: {precisions_to_run}")
    
    # 运行实验
    all_results = {}
    
    for precision in precisions_to_run:
        config = precision_configs[precision]
        print(f"\n{'='*60}")
        print(f"开始 {precision.upper()} 实验")
        print(f"描述: {config['description']}")
        print(f"{'='*60}")
        
        try:
            result = run_single_precision_training(
                precision=precision,
                batch_size=config['batch_size'],
                epochs=args.epochs
            )
            all_results[precision] = result
            
            # 保存中间结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = f"precision_results_{timestamp}.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"   💾 结果已保存: {result_file}")
            
        except Exception as e:
            print(f"   ❌ {precision.upper()} 实验失败: {e}")
            all_results[precision] = {
                'precision': precision,
                'success': False,
                'error': str(e)
            }
    
    # 生成最终报告
    print(f"\n{'='*60}")
    print(f"🎉 精度对比实验完成!")
    print(f"{'='*60}")
    
    successful_experiments = {k: v for k, v in all_results.items() if v.get('success', False)}
    
    if successful_experiments:
        print(f"✅ 成功实验: {len(successful_experiments)}/{len(all_results)}")
        print(f"\n⏱️ 训练时间对比:")
        
        for precision, result in successful_experiments.items():
            print(f"   {precision.upper()}: {result['total_time_hours']:.2f} 小时")
        
        # 计算加速比
        if 'fp32' in successful_experiments:
            fp32_time = successful_experiments['fp32']['total_time_hours']
            print(f"\n🚀 相对FP32的加速比:")
            
            for precision, result in successful_experiments.items():
                if precision != 'fp32':
                    speedup = fp32_time / result['total_time_hours']
                    print(f"   {precision.upper()}: {speedup:.2f}x")
        
        print(f"\n📁 模型保存位置:")
        for precision, result in successful_experiments.items():
            if result.get('model_path'):
                print(f"   {precision.upper()}: {result['model_path']}")
    else:
        print(f"❌ 没有成功的实验")
    
    print(f"\n🎯 下一步: 运行报告质量评估")
    print(f"   使用命令: python evaluate_report_quality.py")


if __name__ == "__main__":
    main()

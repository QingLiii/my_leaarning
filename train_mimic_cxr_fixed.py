#!/usr/bin/env python3
"""
MIMIC-CXR训练脚本 - 基于修复后的R2Gen代码
使用FP32精度确保最佳质量，严格遵循论文配置
"""

import os
import sys
import torch
import argparse
import numpy as np
import wandb
from datetime import datetime
import json
import time

# 添加R2Gen路径
sys.path.append('R2Gen-main')

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel

def setup_wandb(args):
    """设置WandB监控"""
    wandb.init(
        project="R2Gen-MIMIC-CXR-Fixed",
        name=f"mimic_cxr_fp32_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "dataset": "MIMIC-CXR",
            "precision": "FP32",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr_ve": args.lr_ve,
            "lr_ed": args.lr_ed,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "max_seq_length": args.max_seq_length,
            "beam_size": args.beam_size,
            "seed": args.seed
        },
        tags=["MIMIC-CXR", "FP32", "Fixed-BatchNorm", "Paper-Config"]
    )

def log_gpu_memory():
    """记录GPU内存使用情况"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
        
        wandb.log({
            "gpu_memory_allocated_gb": memory_allocated,
            "gpu_memory_reserved_gb": memory_reserved,
            "gpu_memory_free_gb": memory_free,
            "gpu_memory_utilization": memory_allocated / (torch.cuda.get_device_properties(0).total_memory / 1024**3) * 100
        })
        
        return memory_allocated, memory_reserved, memory_free
    return 0, 0, 0

def parse_args():
    parser = argparse.ArgumentParser(description='MIMIC-CXR训练 - 修复版R2Gen')

    # 数据设置
    parser.add_argument('--image_dir', type=str, default='R2Gen-main/data/mimic_cxr/images/',
                        help='图像目录路径')
    parser.add_argument('--ann_path', type=str, default='R2Gen-main/data/mimic_cxr/annotation.json',
                        help='标注文件路径')
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr',
                        help='数据集名称')

    # 视觉提取器设置
    parser.add_argument('--visual_extractor', type=str, default='resnet101',
                        help='视觉特征提取器')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='是否使用预训练的视觉提取器')

    # 模型设置 - 严格按照论文配置
    parser.add_argument('--d_model', type=int, default=512,
                        help='模型维度')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Transformer层数')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='前馈网络维度')
    parser.add_argument('--d_vf', type=int, default=2048,
                        help='视觉特征维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Transformer dropout率')
    parser.add_argument('--logit_layers', type=int, default=1,
                        help='logit层数')

    # RelationalMemory设置
    parser.add_argument('--rm_num_slots', type=int, default=3,
                        help='RelationalMemory槽位数')
    parser.add_argument('--rm_num_heads', type=int, default=8,
                        help='RelationalMemory注意力头数')
    parser.add_argument('--rm_d_model', type=int, default=512,
                        help='RelationalMemory维度')

    # 特殊token设置
    parser.add_argument('--bos_idx', type=int, default=0,
                        help='开始token索引')
    parser.add_argument('--eos_idx', type=int, default=0,
                        help='结束token索引')
    parser.add_argument('--pad_idx', type=int, default=0,
                        help='填充token索引')

    # 采样设置
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='采样方法')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='束搜索大小')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='采样温度')
    parser.add_argument('--sample_n', type=int, default=1,
                        help='每张图像的采样数')
    parser.add_argument('--group_size', type=int, default=1,
                        help='组大小')
    parser.add_argument('--output_logsoftmax', type=int, default=1,
                        help='是否输出log softmax')
    parser.add_argument('--decoding_constraint', type=int, default=0,
                        help='是否使用解码约束')
    parser.add_argument('--block_trigrams', type=int, default=1,
                        help='是否阻止三元组重复')

    # 训练设置 - 严格按照论文配置
    parser.add_argument('--lr_ve', type=float, default=5e-5,
                        help='Visual Encoder学习率')
    parser.add_argument('--lr_ed', type=float, default=1e-4,
                        help='Encoder-Decoder学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='权重衰减')
    parser.add_argument('--amsgrad', type=bool, default=True,
                        help='是否使用AMSGrad')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='优化器类型')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        help='学习率调度器类型')
    parser.add_argument('--step_size', type=int, default=1,
                        help='学习率衰减步长')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='学习率衰减因子')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')

    # 序列设置
    parser.add_argument('--max_seq_length', type=int, default=100,
                        help='最大序列长度')
    parser.add_argument('--threshold', type=int, default=10,
                        help='词频阈值')

    # 训练器设置
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='GPU数量')
    parser.add_argument('--save_period', type=int, default=1,
                        help='保存周期')
    parser.add_argument('--monitor_mode', type=str, default='max',
                        help='监控模式')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4',
                        help='监控指标')
    parser.add_argument('--early_stop', type=int, default=50,
                        help='早停耐心值')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='记录目录')

    # 其他设置
    parser.add_argument('--seed', type=int, default=456789,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='results/mimic_cxr_fixed',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--use_bn', type=int, default=1,
                        help='是否使用BatchNorm')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                        help='语言模型dropout概率')

    return parser.parse_args()

def validate_environment():
    """验证训练环境"""
    print("🔍 验证训练环境...")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 检查数据文件
    args = parse_args()
    if not os.path.exists(args.ann_path):
        print(f"❌ 标注文件不存在: {args.ann_path}")
        return False
    
    if not os.path.exists(args.image_dir):
        print(f"❌ 图像目录不存在: {args.image_dir}")
        return False
    
    print("✅ 数据文件检查通过")
    
    # 检查修复的BatchNorm
    try:
        from modules.att_model import FixedBatchNorm1dWrapper
        print("✅ BatchNorm修复已应用")
    except ImportError:
        print("❌ BatchNorm修复未找到")
        return False
    
    return True

def create_model_and_tokenizer(args):
    """创建模型和分词器"""
    print("🏗️ 创建模型和分词器...")
    
    # 创建分词器
    tokenizer = Tokenizer(args)
    
    # 创建模型
    model = R2GenModel(args, tokenizer)
    
    # 移动到GPU
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ 模型创建完成")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    return model, tokenizer

def create_data_loaders(args, tokenizer):
    """创建数据加载器"""
    print("📊 创建数据加载器...")
    
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    
    print(f"✅ 数据加载器创建完成")
    print(f"   训练集: {len(train_dataloader)} 批次")
    print(f"   验证集: {len(val_dataloader)} 批次")
    print(f"   测试集: {len(test_dataloader)} 批次")
    
    return train_dataloader, val_dataloader, test_dataloader

def create_optimizer_and_scheduler(args, model):
    """创建优化器和学习率调度器"""
    print("⚙️ 创建优化器和调度器...")
    
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    
    print(f"✅ 优化器创建完成")
    print(f"   Visual Encoder LR: {args.lr_ve}")
    print(f"   Encoder-Decoder LR: {args.lr_ed}")
    print(f"   衰减策略: 每{args.step_size}个epoch乘以{args.gamma}")
    
    return optimizer, lr_scheduler

def train_epoch(model, dataloader, optimizer, criterion, epoch, args):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    print(f"🚀 开始第{epoch}个epoch训练...")
    start_time = time.time()
    
    for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(dataloader):
        if torch.cuda.is_available():
            images = images.cuda()
            reports_ids = reports_ids.cuda()
            reports_masks = reports_masks.cuda()
        
        optimizer.zero_grad()
        
        # 前向传播
        output = model(images, reports_ids, mode='train')
        loss = criterion(output, reports_ids, reports_masks)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 记录训练进度
        if batch_idx % 10 == 0:
            current_lr_ve = optimizer.param_groups[0]['lr']
            current_lr_ed = optimizer.param_groups[1]['lr']
            
            wandb.log({
                "train_loss": loss.item(),
                "lr_ve": current_lr_ve,
                "lr_ed": current_lr_ed,
                "epoch": epoch,
                "batch": batch_idx
            })
            
            # 记录GPU内存
            mem_alloc, mem_reserved, mem_free = log_gpu_memory()
            
            print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                  f"Loss: {loss.item():.4f}, "
                  f"LR_VE: {current_lr_ve:.2e}, LR_ED: {current_lr_ed:.2e}, "
                  f"GPU: {mem_alloc:.1f}GB")
    
    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time
    
    print(f"✅ Epoch {epoch} 训练完成")
    print(f"   平均Loss: {avg_loss:.4f}")
    print(f"   训练时间: {epoch_time:.1f}秒")
    
    return avg_loss

def evaluate_model(model, dataloader, tokenizer, split_name):
    """评估模型"""
    print(f"📊 评估模型 ({split_name})...")
    
    model.eval()
    reports_generated = []
    reports_ground_truth = []
    
    with torch.no_grad():
        for images_id, images, reports_ids, reports_masks in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
            
            # 生成报告
            output = model(images, mode='sample')
            reports = model.tokenizer.decode_batch(output.cpu().numpy())
            ground_truths = model.tokenizer.decode_batch(reports_ids.numpy())
            
            reports_generated.extend(reports)
            reports_ground_truth.extend(ground_truths)
    
    # 计算评估指标
    gts = {i: [gt] for i, gt in enumerate(reports_ground_truth)}
    res = {i: [gen] for i, gen in enumerate(reports_generated)}
    scores = compute_scores(gts, res)
    
    print(f"✅ {split_name}评估完成:")
    for metric, score in scores.items():
        print(f"   {metric}: {score:.4f}")
    
    return scores, reports_generated[:5]  # 返回前5个生成样本

def main():
    """主训练函数"""
    print("🚀 开始MIMIC-CXR训练实验...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 验证环境
    if not validate_environment():
        print("❌ 环境验证失败，实验终止")
        return
    
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置WandB
    setup_wandb(args)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建模型和分词器
    model, tokenizer = create_model_and_tokenizer(args)
    
    # 创建数据加载器
    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(args, tokenizer)
    
    # 创建优化器和调度器
    optimizer, lr_scheduler = create_optimizer_and_scheduler(args, model)
    
    # 创建损失函数
    criterion = compute_loss
    
    print(f"\n🎯 开始训练 - {args.epochs} epochs")
    print("="*60)
    
    best_val_score = 0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n📅 Epoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, epoch, args)
        
        # 验证
        val_scores, val_samples = evaluate_model(model, val_dataloader, tokenizer, 'validation')
        
        # 更新学习率
        lr_scheduler.step()
        
        # 记录到WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_scores.items()}
        })
        
        # 保存最佳模型
        current_score = val_scores.get('BLEU_4', 0)
        if current_score > best_val_score:
            best_val_score = current_score
            best_epoch = epoch
            
            # 保存模型
            model_path = os.path.join(args.save_dir, 'model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_scores': val_scores,
                'args': args
            }, model_path)
            
            print(f"🏆 新的最佳模型! BLEU_4: {current_score:.4f}")
        
        print(f"当前最佳: Epoch {best_epoch}, BLEU_4: {best_val_score:.4f}")
    
    print("\n🎉 训练完成!")
    print("="*60)
    
    # 最终测试
    print("📊 最终测试评估...")
    
    # 加载最佳模型
    best_model_path = os.path.join(args.save_dir, 'model_best.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试评估
    test_scores, test_samples = evaluate_model(model, test_dataloader, tokenizer, 'test')
    
    # 记录最终结果
    wandb.log({
        **{f"final_test_{k}": v for k, v in test_scores.items()},
        "best_epoch": best_epoch,
        "best_val_bleu4": best_val_score
    })
    
    # 保存结果
    results = {
        'args': vars(args),
        'best_epoch': best_epoch,
        'best_val_scores': checkpoint['val_scores'],
        'final_test_scores': test_scores,
        'test_samples': test_samples,
        'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = os.path.join(args.save_dir, 'training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存到: {results_path}")
    print(f"✅ 最佳模型已保存到: {best_model_path}")
    
    # 显示最终结果
    print(f"\n🏆 最终结果:")
    print(f"最佳验证BLEU_4: {best_val_score:.4f} (Epoch {best_epoch})")
    print(f"最终测试结果:")
    for metric, score in test_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    main()

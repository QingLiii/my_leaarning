#!/usr/bin/env python3
"""
WandB集成测试脚本
验证WandB监控系统是否正常工作
"""

import sys
import os
import argparse
import torch
import numpy as np

# 添加R2Gen路径
sys.path.append('R2Gen-main')

from modules.wandb_logger import WandBLogger
from modules.enhanced_trainer import EnhancedTrainer
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.loss import compute_loss
from models.r2gen import R2GenModel


def create_test_args():
    """创建测试用的参数"""
    class TestArgs:
        def __init__(self):
            # 数据相关
            self.image_dir = 'R2Gen-main/data/iu_xray/images/'
            self.ann_path = 'R2Gen-main/data/iu_xray/annotation.json'
            self.dataset_name = 'iu_xray'
            self.max_seq_length = 60
            self.threshold = 3
            
            # 训练相关
            self.batch_size = 2  # 小batch size用于测试
            self.epochs = 2      # 只训练2个epoch用于测试
            self.seed = 9223
            
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
            
            # 训练设置
            self.n_gpu = 1
            self.save_dir = 'results/wandb_test'
            self.save_period = 1
            self.monitor_mode = 'max'
            self.monitor_metric = 'BLEU_4'
            self.early_stop = 50
            self.resume = None
            
            # 优化相关（新增）
            self.gradient_accumulation_steps = 1
            self.mixed_precision = None  # 'fp16', 'fp8', None
            self.log_interval = 5  # 更频繁的日志记录用于测试
    
    return TestArgs()


def install_wandb_dependencies():
    """安装WandB相关依赖"""
    try:
        import wandb
        import GPUtil
        print("✅ WandB和监控依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行以下命令安装:")
        print("pip install wandb GPUtil nvidia-ml-py3")
        return False


def test_wandb_logger():
    """测试WandB Logger基础功能"""
    print("\n🧪 测试WandB Logger基础功能...")
    
    try:
        # 创建logger
        logger = WandBLogger(project_name="R2Gen-Test")
        
        # 测试配置
        test_config = {
            'test_mode': True,
            'batch_size': 2,
            'learning_rate': 0.001,
            'model': 'R2Gen'
        }
        
        # 初始化运行
        logger.init_run(test_config, run_name="wandb_logger_test")
        
        # 测试系统监控
        logger.log_system_metrics(force=True)
        
        # 测试训练指标记录
        for i in range(5):
            logger.log_training_metrics(
                epoch=1,
                batch_idx=i,
                loss=3.0 - i * 0.1,
                learning_rate=0.001,
                gradient_norm=0.5
            )
        
        # 测试验证指标记录
        val_metrics = {
            'loss': 2.5,
            'BLEU_4': 0.15,
            'METEOR': 0.20
        }
        logger.log_validation_metrics(1, val_metrics)
        
        # 结束运行
        logger.finish()
        
        print("✅ WandB Logger测试通过")
        return True
        
    except Exception as e:
        print(f"❌ WandB Logger测试失败: {e}")
        return False


def test_enhanced_trainer():
    """测试增强版Trainer"""
    print("\n🧪 测试增强版Trainer...")
    
    try:
        # 创建参数
        args = create_test_args()
        
        # 设置随机种子
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        
        # 创建tokenizer
        tokenizer = Tokenizer(args)
        
        # 创建数据加载器（小数据集用于测试）
        train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
        val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
        test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
        
        print(f"📊 数据加载完成: train={len(train_dataloader)}, val={len(val_dataloader)}, test={len(test_dataloader)}")
        
        # 创建模型
        model = R2GenModel(args, tokenizer)
        print(f"🤖 模型创建完成: {model}")
        
        # 创建优化器和调度器
        optimizer = build_optimizer(args, model)
        lr_scheduler = build_lr_scheduler(args, optimizer)
        
        # 创建WandB logger
        wandb_logger = WandBLogger(project_name="R2Gen-Enhanced-Trainer-Test")
        
        # 创建增强版trainer
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
        
        print("🚀 开始训练测试...")
        
        # 运行训练
        trainer.train()
        
        print("✅ 增强版Trainer测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 增强版Trainer测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='WandB集成测试')
    parser.add_argument('--test-logger', action='store_true', help='测试WandB Logger')
    parser.add_argument('--test-trainer', action='store_true', help='测试增强版Trainer')
    parser.add_argument('--test-all', action='store_true', help='运行所有测试')
    
    args = parser.parse_args()
    
    print("🧪 WandB集成测试开始...")
    print("=" * 50)
    
    # 检查依赖
    if not install_wandb_dependencies():
        return
    
    success_count = 0
    total_tests = 0
    
    # 测试WandB Logger
    if args.test_logger or args.test_all:
        total_tests += 1
        if test_wandb_logger():
            success_count += 1
    
    # 测试增强版Trainer
    if args.test_trainer or args.test_all:
        total_tests += 1
        if test_enhanced_trainer():
            success_count += 1
    
    # 如果没有指定测试，默认运行Logger测试
    if not any([args.test_logger, args.test_trainer, args.test_all]):
        total_tests += 1
        if test_wandb_logger():
            success_count += 1
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print(f"🎯 测试完成: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！WandB集成准备就绪")
        print("\n📋 下一步:")
        print("1. 运行完整训练验证WandB监控")
        print("2. 开始显存优化模块开发")
        print("3. 准备精度对比实验")
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
    
    return success_count == total_tests


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

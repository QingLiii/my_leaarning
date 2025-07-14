#!/usr/bin/env python3
"""
完整的R2Gen精度对比实验
严格按照论文"Generating Radiology Reports via Memory-driven Transformer"配置
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
# from modules.batch_size_optimizer import BatchSizeOptimizer  # 暂时不使用

class PrecisionExperimentConfig:
    """精度实验配置类 - 严格按照论文配置"""
    
    def __init__(self, precision, epochs=15):
        # 基础配置
        self.precision = precision
        self.epochs = epochs
        
        # 数据集相关
        self.image_dir = 'datasets/iu_xray/images/'
        self.ann_path = 'datasets/iu_xray/annotation.json'
        self.dataset_name = 'iu_xray'
        self.max_seq_length = 60
        self.threshold = 3
        self.num_workers = 2
        self.batch_size = 12  # 默认batch size，会被动态调整
        
        # 训练相关 - 按论文配置
        self.seed = 9223
        self.n_gpu = 1
        self.save_period = 1  # 每个epoch保存
        self.monitor_mode = 'max'
        self.monitor_metric = 'BLEU_4'
        self.early_stop = 50  # 允许更多epoch
        self.resume = None
        self.validate_every = 1  # 每个epoch验证
        
        # 优化器相关 - 严格按论文配置
        self.optim = 'Adam'
        self.lr_ve = 5e-5      # visual extractor学习率
        self.lr_ed = 1e-4      # encoder-decoder学习率
        self.weight_decay = 0
        self.amsgrad = True
        
        # 学习率调度 - 按论文要求每epoch衰减0.8
        self.lr_scheduler = 'StepLR'
        self.step_size = 1     # 每个epoch衰减
        self.gamma = 0.8       # 衰减因子0.8
        
        # 模型相关 - 严格按论文Table 3配置
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
        
        # 记忆模块 - 按论文配置
        self.rm_num_slots = 3
        self.rm_num_heads = 8
        self.rm_d_model = 512
        
        # 视觉特征提取
        self.visual_extractor = 'resnet101'
        self.visual_extractor_pretrained = True
        
        # 采样相关 - 按论文配置
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
        self.save_dir = f'results/precision_comparison/{precision}'
        self.record_dir = f'results/precision_comparison/{precision}'
        self.experiment_name = f'R2Gen_{precision}_precision'
        
        # 其他必需参数
        self.drop_prob_lm = 0.5

def run_precision_experiment(precision, epochs=15):
    """运行单个精度实验"""
    
    print(f"\n{'='*60}")
    print(f"开始 {precision.upper()} 实验")
    print(f"描述: {get_precision_description(precision)}")
    print(f"{'='*60}")
    
    # 创建配置
    args = PrecisionExperimentConfig(precision, epochs)
    
    # 根据精度设置batch size
    if precision == 'fp32':
        batch_size = 12  # 基准batch size
    elif precision == 'fp16':
        batch_size = 16  # FP16可以使用更大batch size
    else:  # fp8
        batch_size = 20  # FP8可以使用最大batch size

    # 更新args中的batch_size
    args.batch_size = batch_size
    
    print(f"\n🚀 开始 {precision.upper()} 精度训练...")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    # 设置API key
    os.environ['WANDB_API_KEY'] = '68c9ce2a167992d06678c4fdc0d1075b5dfff922'
    wandb_logger = WandBLogger(project_name="R2Gen-Complete-Precision-Comparison")
    
    wandb_config = {
        'precision': precision,
        'mixed_precision': args.mixed_precision,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate_ve': args.lr_ve,
        'learning_rate_ed': args.lr_ed,
        'lr_scheduler': args.lr_scheduler,
        'lr_decay_factor': args.gamma,
        'lr_decay_step': args.step_size,
        'model': 'R2Gen',
        'dataset': args.dataset_name,
        'experiment_type': 'complete_precision_comparison',
        'optimizer': args.optim,
        'rm_num_slots': args.rm_num_slots,
        'rm_d_model': args.rm_d_model,
        'beam_size': args.beam_size,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads
    }
    
    run_name = f"R2Gen_{precision}_complete_bs{batch_size}_ep{epochs}"
    wandb_logger.init_run(wandb_config, run_name=run_name)
    
    # 创建训练器
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
    
    # 开始训练
    print(f"   🏃 开始训练...")
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = (time.time() - start_time) / 3600  # 转换为小时
        
        print(f"   ✅ {precision.upper()} 训练成功完成!")
        wandb_logger.finish()
        
        return {
            'precision': precision,
            'status': 'success',
            'training_time_hours': training_time,
            'batch_size': batch_size,
            'epochs': epochs,
            'model_path': f'{args.save_dir}/model_best.pth'
        }
        
    except Exception as e:
        print(f"   ❌ {precision.upper()} 训练失败: {str(e)}")
        wandb_logger.finish()
        
        return {
            'precision': precision,
            'status': 'failed',
            'error': str(e),
            'batch_size': batch_size,
            'epochs': epochs
        }

def get_precision_description(precision):
    """获取精度描述"""
    descriptions = {
        'fp32': 'Full Precision (FP32)',
        'fp16': 'Half Precision (FP16) with Mixed Precision Training',
        'fp8': 'Quarter Precision (FP8) with Mixed Precision Training'
    }
    return descriptions.get(precision, 'Unknown Precision')

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='完整R2Gen精度对比实验')
    parser.add_argument('--epochs', type=int, default=15, help='训练epoch数')
    parser.add_argument('--precisions', nargs='+', default=['fp8', 'fp16', 'fp32'], 
                       help='要测试的精度列表')
    
    args = parser.parse_args()
    
    print("🚀 R2Gen完整精度对比实验")
    print(f"   精度范围: {', '.join(args.precisions)}")
    print(f"   训练epochs: {args.epochs}")
    print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   将运行精度: {args.precisions}")
    
    # 验证前置条件
    print(f"\n📋 验证前置条件...")
    
    # 验证tokenizer解码
    sys.path.append('R2Gen-main')
    from modules.tokenizers import Tokenizer
    import argparse as arg_ns
    
    test_args = arg_ns.Namespace()
    test_args.ann_path = 'datasets/iu_xray/annotation.json'
    test_args.threshold = 3
    test_args.dataset_name = 'iu_xray'
    
    tokenizer = Tokenizer(test_args)
    test_text = 'the lungs are clear'
    encoded = tokenizer(test_text)
    decoded = tokenizer.decode(encoded)
    
    if len(decoded) == 0:
        print("❌ Tokenizer解码失败，实验终止")
        return
    else:
        print("✅ Tokenizer解码正常")
    
    # 执行实验
    results = []
    
    for precision in args.precisions:
        result = run_precision_experiment(precision, args.epochs)
        results.append(result)
        
        # 检查BLEU分数合理性
        if result['status'] == 'success':
            print(f"✅ {precision.upper()} 实验成功")
        else:
            print(f"❌ {precision.upper()} 实验失败: {result.get('error', 'Unknown error')}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'complete_precision_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 总结
    print(f"\n{'='*60}")
    print(f"🎉 完整精度对比实验完成!")
    print(f"{'='*60}")
    
    successful_experiments = [r for r in results if r['status'] == 'success']
    failed_experiments = [r for r in results if r['status'] == 'failed']
    
    print(f"✅ 成功实验: {len(successful_experiments)}/{len(results)}")
    
    if successful_experiments:
        print(f"\n⏱️ 训练时间对比:")
        for result in successful_experiments:
            print(f"   {result['precision'].upper()}: {result['training_time_hours']:.2f} 小时")
        
        # 计算加速比
        fp32_time = next((r['training_time_hours'] for r in successful_experiments if r['precision'] == 'fp32'), None)
        if fp32_time:
            print(f"\n🚀 相对FP32的加速比:")
            for result in successful_experiments:
                if result['precision'] != 'fp32':
                    speedup = (fp32_time - result['training_time_hours']) / fp32_time * 100
                    print(f"   {result['precision'].upper()}: {speedup:.1f}% 更快")
    
    if failed_experiments:
        print(f"\n❌ 失败实验:")
        for result in failed_experiments:
            print(f"   {result['precision'].upper()}: {result.get('error', 'Unknown error')}")
    
    print(f"\n📁 模型保存位置:")
    for result in successful_experiments:
        print(f"   {result['precision'].upper()}: {result['model_path']}")
    
    print(f"\n📊 结果文件: {results_file}")
    print(f"\n🎯 下一步: 生成完整HTML报告")
    print(f"   使用命令: python generate_complete_precision_report.py")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
精度对比实验 + 报告质量评估
完整的FP32 vs FP16 vs FP8对比，包含定量指标和定性报告分析
"""

import sys
import os
import argparse
import torch
import numpy as np
import json
from datetime import datetime

# 添加路径
sys.path.append('R2Gen-main')

from modules.wandb_logger import WandBLogger
from modules.memory_monitor import MemoryMonitor
from modules.enhanced_trainer import EnhancedTrainer
from modules.report_quality_evaluator import ReportQualityEvaluator
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.loss import compute_loss
from models.r2gen import R2GenModel


class PrecisionComparisonExperiment:
    """
    精度对比实验管理器
    """
    
    def __init__(self, base_args):
        self.base_args = base_args
        self.results = {}
        self.trained_models = {}
        self.wandb_logger = WandBLogger(project_name="R2Gen-Precision-Comparison")
        self.memory_monitor = MemoryMonitor()
        
        # 精度配置
        self.precision_configs = {
            'fp32': {
                'mixed_precision': None,
                'description': 'Full Precision (FP32)',
                'expected_speedup': 1.0
            },
            'fp16': {
                'mixed_precision': 'fp16',
                'description': 'Half Precision (FP16)',
                'expected_speedup': 1.5
            },
            'fp8': {
                'mixed_precision': 'fp8',
                'description': 'FP8 Precision (experimental)',
                'expected_speedup': 2.0
            }
        }
        
        print(f"✅ 精度对比实验初始化完成")
        print(f"   对比精度: {list(self.precision_configs.keys())}")
        print(f"   训练epochs: {base_args.epochs}")
    
    def run_single_precision_experiment(self, precision: str) -> Dict:
        """
        运行单个精度的实验
        
        Args:
            precision: 精度类型 ('fp32', 'fp16', 'fp8')
            
        Returns:
            实验结果字典
        """
        config = self.precision_configs[precision]
        print(f"\n🚀 开始 {precision.upper()} 精度实验...")
        print(f"   描述: {config['description']}")
        
        # 创建实验参数
        args = self._create_experiment_args(precision, config)
        
        # 设置随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # 创建数据加载器
        tokenizer = Tokenizer(args)
        train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
        val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
        test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
        
        # 创建模型
        model = R2GenModel(args, tokenizer)
        
        # 创建优化器
        optimizer = build_optimizer(args, model)
        lr_scheduler = build_lr_scheduler(args, optimizer)
        
        # 初始化WandB运行
        wandb_config = {
            'precision': precision,
            'mixed_precision': config['mixed_precision'],
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate_ve': args.lr_ve,
            'learning_rate_ed': args.lr_ed,
            'model': 'R2Gen',
            'dataset': args.dataset_name,
            'experiment_type': 'precision_comparison'
        }
        
        run_name = f"R2Gen_{precision}_precision_{args.epochs}epochs"
        self.wandb_logger.init_run(wandb_config, run_name=run_name)
        
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
            wandb_logger=self.wandb_logger,
            enable_wandb=True
        )
        
        # 记录实验开始
        start_time = datetime.now()
        print(f"   开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 运行训练
        try:
            trainer.train()
            training_success = True
        except Exception as e:
            print(f"❌ {precision} 训练失败: {e}")
            training_success = False
            return {'precision': precision, 'success': False, 'error': str(e)}
        
        # 记录实验结束
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print(f"   结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   总耗时: {total_time/3600:.2f} 小时")
        
        # 收集结果
        result = {
            'precision': precision,
            'success': training_success,
            'total_time_hours': total_time / 3600,
            'total_time_seconds': total_time,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'final_metrics': trainer.best_recorder,
            'model_path': os.path.join(args.save_dir, 'model_best.pth'),
            'config': config
        }
        
        # 保存训练好的模型
        self.trained_models[precision] = model
        
        # 结束WandB运行
        self.wandb_logger.finish()
        
        print(f"✅ {precision.upper()} 实验完成")
        return result
    
    def _create_experiment_args(self, precision: str, config: Dict):
        """创建实验参数"""
        import copy
        args = copy.deepcopy(self.base_args)
        
        # 设置精度相关参数
        args.mixed_precision = config['mixed_precision']
        args.save_dir = f"results/precision_comparison/{precision}"
        args.experiment_name = f"R2Gen_{precision}_precision"
        
        # 根据精度调整batch size (利用显存优化结果)
        if precision == 'fp32':
            args.batch_size = 12  # 基于之前的优化结果
        elif precision == 'fp16':
            args.batch_size = 24  # FP16可以用更大的batch size
        else:  # fp8
            args.batch_size = 32  # FP8理论上可以用最大的batch size
        
        # 确保目录存在
        os.makedirs(args.save_dir, exist_ok=True)
        
        return args
    
    def run_all_precision_experiments(self) -> Dict:
        """
        运行所有精度的实验
        
        Returns:
            所有实验结果
        """
        print(f"\n🎯 开始完整的精度对比实验")
        print(f"=" * 60)
        
        all_results = {}
        
        for precision in self.precision_configs.keys():
            try:
                result = self.run_single_precision_experiment(precision)
                all_results[precision] = result
                
                # 保存中间结果
                self._save_intermediate_results(all_results)
                
            except Exception as e:
                print(f"❌ {precision} 实验失败: {e}")
                all_results[precision] = {
                    'precision': precision,
                    'success': False,
                    'error': str(e)
                }
        
        self.results = all_results
        return all_results
    
    def evaluate_report_quality(self, num_samples: int = 5) -> Dict:
        """
        评估报告质量
        
        Args:
            num_samples: 评估样本数量
            
        Returns:
            质量评估结果
        """
        if not self.trained_models:
            raise ValueError("请先运行训练实验")
        
        print(f"\n📝 开始报告质量评估...")
        
        # 准备测试数据
        args = self.base_args
        tokenizer = Tokenizer(args)
        test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
        
        # 收集测试样本
        test_images = []
        test_image_ids = []
        ground_truth_reports = []
        
        for i, (image_ids, images, reports_ids, reports_masks) in enumerate(test_dataloader):
            if len(test_images) >= num_samples:
                break
            
            for j in range(min(images.size(0), num_samples - len(test_images))):
                test_images.append(images[j])
                test_image_ids.append(image_ids[j])
                
                # 解码真实报告
                report_tokens = reports_ids[j].cpu().numpy()
                report_text = tokenizer.decode_batch([report_tokens])[0]
                ground_truth_reports.append(report_text)
        
        print(f"   收集了 {len(test_images)} 个测试样本")
        
        # 创建质量评估器
        evaluator = ReportQualityEvaluator(
            models_dict=self.trained_models,
            tokenizer=tokenizer
        )
        
        # 生成样本报告
        sample_reports = evaluator.generate_sample_reports(
            test_images=test_images,
            test_image_ids=test_image_ids,
            ground_truth_reports=ground_truth_reports,
            num_samples=num_samples
        )
        
        # 分析质量
        quality_analysis = evaluator.analyze_report_quality()
        
        # 生成对比报告
        report_path = f"precision_comparison_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        evaluator.generate_comparison_report(report_path)
        
        # 打印摘要
        evaluator.print_summary()
        
        return {
            'sample_reports': sample_reports,
            'quality_analysis': quality_analysis,
            'report_path': report_path,
            'evaluator': evaluator
        }
    
    def _save_intermediate_results(self, results: Dict):
        """保存中间结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"precision_comparison_results_{timestamp}.json"
        
        # 转换为可序列化的格式
        serializable_results = {}
        for precision, result in results.items():
            if result.get('success', False):
                serializable_results[precision] = {
                    'precision': result['precision'],
                    'success': result['success'],
                    'total_time_hours': result['total_time_hours'],
                    'start_time': result['start_time'],
                    'end_time': result['end_time'],
                    # 注意：final_metrics可能包含不可序列化的对象
                    'config': result['config']
                }
            else:
                serializable_results[precision] = result
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 中间结果已保存: {filename}")
    
    def generate_final_report(self):
        """生成最终报告"""
        if not self.results:
            print("❌ 没有实验结果可生成报告")
            return
        
        print(f"\n📊 精度对比实验最终报告")
        print(f"=" * 60)
        
        # 成功的实验
        successful_experiments = {k: v for k, v in self.results.items() if v.get('success', False)}
        
        if not successful_experiments:
            print("❌ 没有成功的实验")
            return
        
        print(f"✅ 成功实验: {len(successful_experiments)}/{len(self.results)}")
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
                    expected = result['config']['expected_speedup']
                    print(f"   {precision.upper()}: {speedup:.2f}x (预期: {expected:.1f}x)")
        
        print(f"\n📁 模型保存位置:")
        for precision, result in successful_experiments.items():
            if 'model_path' in result:
                print(f"   {precision.upper()}: {result['model_path']}")


def create_base_args():
    """创建基础参数"""
    class Args:
        def __init__(self):
            # 数据相关
            self.image_dir = 'data/iu_xray/images/'
            self.ann_path = 'data/iu_xray/annotation.json'
            self.dataset_name = 'iu_xray'
            self.max_seq_length = 60
            self.threshold = 3
            
            # 训练相关
            self.epochs = 15  # 精度对比实验用15个epoch
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
            self.save_period = 5
            self.monitor_mode = 'max'
            self.monitor_metric = 'BLEU_4'
            self.early_stop = 50
            self.resume = None
            
            # 优化相关
            self.gradient_accumulation_steps = 1
            self.log_interval = 50
    
    return Args()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='R2Gen精度对比实验')
    parser.add_argument('--epochs', type=int, default=15, help='训练epochs数')
    parser.add_argument('--quality-eval', action='store_true', help='运行报告质量评估')
    parser.add_argument('--quality-samples', type=int, default=5, help='质量评估样本数')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'fp8', 'all'], 
                       default='all', help='运行特定精度实验')
    
    args = parser.parse_args()
    
    print(f"🚀 R2Gen精度对比实验")
    print(f"   训练epochs: {args.epochs}")
    print(f"   精度范围: {args.precision}")
    print(f"   质量评估: {'是' if args.quality_eval else '否'}")
    
    # 创建基础参数
    base_args = create_base_args()
    base_args.epochs = args.epochs
    
    # 创建实验管理器
    experiment = PrecisionComparisonExperiment(base_args)
    
    # 运行实验
    if args.precision == 'all':
        results = experiment.run_all_precision_experiments()
    else:
        results = {args.precision: experiment.run_single_precision_experiment(args.precision)}
    
    # 生成最终报告
    experiment.generate_final_report()
    
    # 运行质量评估
    if args.quality_eval and any(r.get('success', False) for r in results.values()):
        print(f"\n🎯 开始报告质量评估...")
        quality_results = experiment.evaluate_report_quality(args.quality_samples)
        print(f"✅ 质量评估完成，报告已保存: {quality_results['report_path']}")
    
    print(f"\n🎉 精度对比实验完成！")


if __name__ == "__main__":
    main()

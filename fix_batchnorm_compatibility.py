#!/usr/bin/env python3
"""
修复BatchNorm兼容性问题
创建支持混合精度训练的版本
"""

import sys
import torch
import torch.nn as nn
import shutil
import os
from datetime import datetime

sys.path.append('R2Gen-main')

class FixedBatchNorm1dWrapper(nn.Module):
    """
    修复的BatchNorm1d包装器
    自动处理输入格式转换，支持混合精度训练
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        
    def forward(self, x):
        """
        自动处理输入格式
        输入: (batch, seq_len, features) 或 (batch, features)
        输出: 与输入相同的格式
        """
        original_shape = x.shape
        
        if len(x.shape) == 3:
            # (batch, seq_len, features) -> (batch, features, seq_len)
            x = x.transpose(1, 2)
            x = self.bn(x)
            # (batch, features, seq_len) -> (batch, seq_len, features)
            x = x.transpose(1, 2)
        else:
            # (batch, features) - 直接处理
            x = self.bn(x)
            
        return x

def create_fixed_att_model():
    """创建修复版本的att_model.py"""
    
    print("🔧 创建修复版本的att_model.py...")
    
    # 读取原始文件
    original_file = 'R2Gen-main/modules/att_model.py'
    backup_file = f'R2Gen-main/modules/att_model_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
    fixed_file = 'R2Gen-main/modules/att_model.py'
    
    # 备份原始文件
    shutil.copy2(original_file, backup_file)
    print(f"✅ 原始文件已备份到: {backup_file}")
    
    # 读取原始内容
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加修复的BatchNorm包装器
    wrapper_code = '''
class FixedBatchNorm1dWrapper(nn.Module):
    """
    修复的BatchNorm1d包装器
    自动处理输入格式转换，支持混合精度训练
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        
    def forward(self, x):
        """
        自动处理输入格式
        输入: (batch, seq_len, features) 或 (batch, features)
        输出: 与输入相同的格式
        """
        if len(x.shape) == 3:
            # (batch, seq_len, features) -> (batch, features, seq_len)
            x = x.transpose(1, 2)
            x = self.bn(x)
            # (batch, features, seq_len) -> (batch, seq_len, features)
            x = x.transpose(1, 2)
        else:
            # (batch, features) - 直接处理
            x = self.bn(x)
        return x

'''
    
    # 在import语句后添加包装器
    import_end = content.find('\\n\\nclass')
    if import_end == -1:
        import_end = content.find('\\nclass')
    
    new_content = content[:import_end] + wrapper_code + content[import_end:]
    
    # 替换BatchNorm1d为FixedBatchNorm1dWrapper
    new_content = new_content.replace(
        'nn.BatchNorm1d(self.att_feat_size)',
        'FixedBatchNorm1dWrapper(self.att_feat_size)'
    )
    new_content = new_content.replace(
        'nn.BatchNorm1d(self.input_encoding_size)',
        'FixedBatchNorm1dWrapper(self.input_encoding_size)'
    )
    
    # 写入修复后的文件
    with open(fixed_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ 修复版本已保存到: {fixed_file}")
    return backup_file

def test_fixed_model():
    """测试修复后的模型"""
    print("\\n🧪 测试修复后的模型...")
    
    try:
        # 重新导入修复后的模块
        import importlib
        if 'modules.att_model' in sys.modules:
            importlib.reload(sys.modules['modules.att_model'])
        if 'models.r2gen' in sys.modules:
            importlib.reload(sys.modules['models.r2gen'])
        
        from modules.tokenizers import Tokenizer
        from models.r2gen import R2GenModel
        import argparse
        
        # 创建测试参数
        args = argparse.Namespace()
        args.ann_path = 'datasets/iu_xray/annotation.json'
        args.threshold = 3
        args.dataset_name = 'iu_xray'
        args.image_dir = 'datasets/iu_xray/images/'
        args.max_seq_length = 60
        args.d_model = 512
        args.d_ff = 512
        args.d_vf = 2048
        args.num_heads = 8
        args.num_layers = 3
        args.dropout = 0.1
        args.logit_layers = 1
        args.bos_idx = 0
        args.eos_idx = 0
        args.pad_idx = 0
        args.use_bn = True
        args.rm_num_slots = 3
        args.rm_num_heads = 8
        args.rm_d_model = 512
        args.visual_extractor = 'resnet101'
        args.visual_extractor_pretrained = True
        args.sample_method = 'beam_search'
        args.beam_size = 3
        args.temperature = 1.0
        args.sample_n = 1
        args.drop_prob_lm = 0.5
        
        # 创建模型
        tokenizer = Tokenizer(args)
        model = R2GenModel(args, tokenizer).cuda()
        print("✅ 修复后的模型创建成功")
        
        # 测试FP32
        print("🧪 测试FP32...")
        batch_size = 2
        images = torch.randn(batch_size, 2, 3, 224, 224).cuda()
        targets = torch.randint(1, 100, (batch_size, 20)).cuda()
        
        with torch.no_grad():
            output = model(images, targets, mode='train')
            print(f"✅ FP32成功: output shape={output.shape}")
        
        # 测试FP16
        print("🧪 测试FP16...")
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                with torch.no_grad():
                    output = model(images, targets, mode='train')
                    print(f"✅ FP16成功: output shape={output.shape}")
        except Exception as e:
            print(f"❌ FP16失败: {e}")
            return False
        
        # 测试推理模式
        print("🧪 测试推理模式...")
        try:
            with torch.no_grad():
                output = model(images, mode='sample')
                print(f"✅ 推理成功: output shape={output.shape}")
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_mixed_precision_trainer():
    """创建支持混合精度的训练器"""
    print("\\n🔧 创建混合精度训练器...")
    
    trainer_code = '''#!/usr/bin/env python3
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
    
    print(f"\\n{'='*60}")
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
    print(f"\\n📊 实验结果: {result}")
'''
    
    # 保存训练器文件
    trainer_file = 'mixed_precision_trainer_fixed.py'
    with open(trainer_file, 'w', encoding='utf-8') as f:
        f.write(trainer_code)
    
    print(f"✅ 混合精度训练器已保存到: {trainer_file}")
    return trainer_file

def main():
    """主函数"""
    print("🚀 开始修复BatchNorm兼容性问题...")
    
    # 创建修复版本
    backup_file = create_fixed_att_model()
    
    # 测试修复后的模型
    success = test_fixed_model()
    
    if success:
        print("\\n✅ BatchNorm兼容性问题修复成功!")
        
        # 创建混合精度训练器
        trainer_file = create_mixed_precision_trainer()
        
        print(f"\\n🎯 下一步:")
        print(f"1. 运行FP32实验: python {trainer_file} --precision fp32 --epochs 15")
        print(f"2. 运行FP16实验: python {trainer_file} --precision fp16 --epochs 15")
        print(f"3. 运行FP8实验: python {trainer_file} --precision fp8 --epochs 15")
        
    else:
        print("\\n❌ 修复失败，恢复原始文件...")
        shutil.copy2(backup_file, 'R2Gen-main/modules/att_model.py')
        print(f"已从备份恢复: {backup_file}")

if __name__ == "__main__":
    main()

"""
增强版Trainer - 集成WandB监控和优化功能
基于原始trainer.py，添加全面的监控和优化功能
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from numpy import inf
from .trainer import BaseTrainer
from .wandb_logger import WandBLogger


class EnhancedTrainer(BaseTrainer):
    """
    增强版训练器
    集成WandB监控、显存优化、混合精度等功能
    """
    
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, 
                 train_dataloader, val_dataloader, test_dataloader, 
                 wandb_logger=None, enable_wandb=True):
        """
        初始化增强版训练器
        
        Args:
            model: 模型
            criterion: 损失函数
            metric_ftns: 评估指标函数
            optimizer: 优化器
            args: 训练参数
            lr_scheduler: 学习率调度器
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            test_dataloader: 测试数据加载器
            wandb_logger: WandB日志记录器
            enable_wandb: 是否启用WandB监控
        """
        super().__init__(model, criterion, metric_ftns, optimizer, args)
        
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # WandB监控
        self.enable_wandb = enable_wandb and wandb_logger is not None
        self.wandb_logger = wandb_logger
        
        # 优化相关参数
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.mixed_precision = getattr(args, 'mixed_precision', None)  # 'fp16', 'fp8', None
        self.log_interval = getattr(args, 'log_interval', 50)  # 日志记录间隔
        
        # 混合精度设置
        self.use_amp = self.mixed_precision in ['fp16', 'fp8']
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print(f"✅ 启用混合精度训练: {self.mixed_precision}")
        
        # 性能统计
        self.batch_times = []
        self.forward_times = []
        self.backward_times = []
        
        # 初始化WandB
        if self.enable_wandb:
            self._init_wandb_monitoring()
    
    def _init_wandb_monitoring(self):
        """初始化WandB监控"""
        try:
            # 准备配置信息
            config = {
                'model_name': 'R2Gen',
                'dataset': getattr(self.args, 'dataset_name', 'unknown'),
                'batch_size': getattr(self.args, 'batch_size', 'unknown'),
                'learning_rate_ve': getattr(self.args, 'lr_ve', 'unknown'),
                'learning_rate_ed': getattr(self.args, 'lr_ed', 'unknown'),
                'epochs': getattr(self.args, 'epochs', 'unknown'),
                'optimizer': getattr(self.args, 'optim', 'unknown'),
                'lr_scheduler': getattr(self.args, 'lr_scheduler', 'unknown'),
                'mixed_precision': self.mixed_precision,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'seed': getattr(self.args, 'seed', 'unknown'),
                'max_seq_length': getattr(self.args, 'max_seq_length', 'unknown'),
                'threshold': getattr(self.args, 'threshold', 'unknown'),
            }
            
            # 生成运行名称
            precision_suffix = f"_{self.mixed_precision}" if self.mixed_precision else "_fp32"
            run_name = f"R2Gen_{config['dataset']}_bs{config['batch_size']}{precision_suffix}"
            
            # 初始化WandB运行
            self.wandb_logger.init_run(
                config=config,
                run_name=run_name,
                tags=["enhanced_trainer", "optimization", self.mixed_precision or "fp32"]
            )
            
            # 记录模型信息
            self.wandb_logger.log_model_info(self.model)
            
            print("✅ WandB监控初始化完成")
            
        except Exception as e:
            print(f"⚠️ WandB初始化失败: {e}")
            self.enable_wandb = False
    
    def _train_epoch(self, epoch):
        """训练一个epoch"""
        if self.enable_wandb:
            self.wandb_logger.start_epoch(epoch)
        
        train_loss = 0
        self.model.train()
        
        epoch_start_time = time.time()
        accumulated_loss = 0
        
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            batch_start_time = time.time()
            
            # 数据移动到设备
            images = images.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)
            
            # 前向传播
            forward_start_time = time.time()
            loss = self._forward_step(images, reports_ids, reports_masks)
            forward_time = time.time() - forward_start_time
            self.forward_times.append(forward_time)
            
            # 梯度累积处理
            loss = loss / self.gradient_accumulation_steps
            accumulated_loss += loss.item()
            
            # 反向传播
            backward_start_time = time.time()
            self._backward_step(loss)
            backward_time = time.time() - backward_start_time
            self.backward_times.append(backward_time)
            
            # 参数更新（每accumulation_steps步更新一次）
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self._optimizer_step()
                
                # 记录训练指标
                if self.enable_wandb:
                    self._log_training_step(epoch, batch_idx, accumulated_loss)
                
                accumulated_loss = 0
            
            train_loss += loss.item() * self.gradient_accumulation_steps
            
            # 记录批次时间
            batch_time = time.time() - batch_start_time
            self.batch_times.append(batch_time)
            
            # 定期打印进度
            if batch_idx % self.log_interval == 0:
                self._print_progress(epoch, batch_idx, loss.item() * self.gradient_accumulation_steps)
        
        # Epoch结束处理
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_loss / len(self.train_dataloader)
        
        if self.enable_wandb:
            self._log_epoch_summary(epoch, epoch_time, avg_train_loss)
            self.wandb_logger.end_epoch(epoch)
        
        return {'train_loss': avg_train_loss}
    
    def _forward_step(self, images, reports_ids, reports_masks):
        """前向传播步骤"""
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                output = self.model(images, reports_ids, mode='train')
                loss = self.criterion(output, reports_ids, reports_masks)
        else:
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)

        return loss
    
    def _backward_step(self, loss):
        """反向传播步骤"""
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _optimizer_step(self):
        """优化器更新步骤"""
        if self.use_amp:
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            
            # 参数更新
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 梯度裁剪
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            
            # 参数更新
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def _log_training_step(self, epoch, batch_idx, loss):
        """记录训练步骤指标"""
        # 计算梯度范数
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # 获取学习率
        lr_ve = self.optimizer.param_groups[0]['lr']
        lr_ed = self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else lr_ve
        
        # 计算性能指标
        avg_batch_time = np.mean(self.batch_times[-10:]) if self.batch_times else 0
        avg_forward_time = np.mean(self.forward_times[-10:]) if self.forward_times else 0
        avg_backward_time = np.mean(self.backward_times[-10:]) if self.backward_times else 0
        
        # 记录到WandB
        self.wandb_logger.log_training_metrics(
            epoch=epoch,
            batch_idx=batch_idx,
            loss=loss,
            learning_rate=lr_ve,
            learning_rate_ve=lr_ve,
            learning_rate_ed=lr_ed,
            gradient_norm=total_norm,
            batch_time=avg_batch_time,
            forward_time=avg_forward_time,
            backward_time=avg_backward_time,
            samples_per_sec=self.args.batch_size / avg_batch_time if avg_batch_time > 0 else 0
        )
    
    def _log_epoch_summary(self, epoch, epoch_time, avg_loss):
        """记录epoch总结"""
        # 计算平均性能指标
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
        avg_forward_time = np.mean(self.forward_times) if self.forward_times else 0
        avg_backward_time = np.mean(self.backward_times) if self.backward_times else 0
        
        # 记录内存使用
        self.wandb_logger.log_memory_usage(f"epoch_{epoch}_end")
        
        # 清理性能统计（保持最近的数据）
        if len(self.batch_times) > 1000:
            self.batch_times = self.batch_times[-500:]
            self.forward_times = self.forward_times[-500:]
            self.backward_times = self.backward_times[-500:]
    
    def _print_progress(self, epoch, batch_idx, loss):
        """打印训练进度"""
        total_batches = len(self.train_dataloader)
        progress = (batch_idx + 1) / total_batches * 100
        
        avg_batch_time = np.mean(self.batch_times[-10:]) if self.batch_times else 0
        samples_per_sec = self.args.batch_size / avg_batch_time if avg_batch_time > 0 else 0
        
        print(f"Epoch {epoch} [{batch_idx+1}/{total_batches} ({progress:.1f}%)] "
              f"Loss: {loss:.6f} | Speed: {samples_per_sec:.1f} samples/sec")
    
    def _evaluate(self, data_loader, epoch, split='val'):
        """评估模型"""
        self.model.eval()
        
        if self.enable_wandb:
            self.wandb_logger.log_memory_usage(f"{split}_start")
        
        with torch.no_grad():
            val_gts, val_res = [], []
            val_loss = 0
            
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(data_loader):
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)
                
                # 前向传播
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        output = self.model(images, reports_ids, mode='train')
                        loss = self.criterion(output, reports_ids, reports_masks)
                else:
                    output = self.model(images, reports_ids, mode='train')
                    loss = self.criterion(output, reports_ids, reports_masks)
                
                val_loss += loss.item()
                
                # 生成报告用于评估
                generated_ids = self.model(images, mode='sample')

                # 创建tokenizer用于解码
                from modules.tokenizers import Tokenizer
                tokenizer = Tokenizer(self.args)

                # 解码生成的报告
                if isinstance(generated_ids, torch.Tensor):
                    # 如果返回的是tensor，解码每个序列
                    generated_reports = tokenizer.decode_batch(generated_ids.cpu().numpy())
                else:
                    # 如果返回的已经是文本列表
                    generated_reports = generated_ids

                val_res.extend(generated_reports)

                # 解码ground truth报告
                gt_reports = tokenizer.decode_batch(reports_ids.cpu().numpy())
                val_gts.extend(gt_reports)
        
        # 计算评估指标
        val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                  {i: [re] for i, re in enumerate(val_res)})
        
        avg_val_loss = val_loss / len(data_loader)
        val_met['loss'] = avg_val_loss
        
        # 记录到WandB
        if self.enable_wandb:
            if split == 'val':
                self.wandb_logger.log_validation_metrics(epoch, val_met)
            else:
                self.wandb_logger.log_test_metrics(epoch, val_met)
            
            self.wandb_logger.log_memory_usage(f"{split}_end")
        
        return val_met
    
    def train(self):
        """主训练循环"""
        print("🚀 开始增强版训练...")
        
        not_improved_count = 0
        validate_every = getattr(self.args, 'validate_every', 1)  # 默认每个epoch验证

        for epoch in range(self.start_epoch, self.epochs + 1):
            # 训练一个epoch
            result = self._train_epoch(epoch)

            # 只在指定的epoch进行验证
            if epoch % validate_every == 0 or epoch == self.epochs:
                print(f"📊 Epoch {epoch}: 开始验证...")

                # 验证
                val_log = self._evaluate(self.val_dataloader, epoch, 'val')
                result.update(**{'val_' + k: v for k, v in val_log.items()})

                # 测试
                test_log = self._evaluate(self.test_dataloader, epoch, 'test')
                result.update(**{'test_' + k: v for k, v in test_log.items()})

                # 记录最佳结果
                log = {'epoch': epoch}
                log.update(result)
                self._record_best(log)

                # 打印结果
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))
            else:
                # 只训练，不验证
                train_loss = result.get('train_loss', result.get('loss', 0))
                log = {'epoch': epoch, 'train_loss': train_loss}
                print(f"\tEpoch {epoch}: train_loss = {train_loss:.6f}")

            # 学习率调度
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # 早停检查（只在验证epoch进行）
            best = False
            if (epoch % validate_every == 0 or epoch == self.epochs) and self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                              (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                # 如果不是每个epoch验证，则禁用早停
                if validate_every == 1 and not_improved_count > self.early_stop:
                    print("Validation performance didn't improve for {} epochs. Training stops.".format(self.early_stop))
                    break
            
            # 保存检查点
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        
        # 训练结束
        self._print_best()
        self._print_best_to_file()
        
        if self.enable_wandb:
            self.wandb_logger.finish()
        
        print("✅ 训练完成！")

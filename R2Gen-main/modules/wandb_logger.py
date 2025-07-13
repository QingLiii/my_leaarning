"""
WandB监控系统 - 全方位训练和系统性能监控
支持训练指标、系统性能、GPU监控等功能
"""

import wandb
import torch
import psutil
import time
import os
import subprocess
from typing import Dict, Any, Optional
import numpy as np
from collections import defaultdict

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPUtil未安装，GPU监控功能将被禁用")

try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("⚠️ nvidia-ml-py3未安装，详细GPU监控功能将被禁用")


class WandBLogger:
    """
    WandB监控系统主类
    提供全方位的训练过程和系统性能监控
    """
    
    def __init__(self, 
                 project_name: str = "R2Gen-Optimization",
                 api_key: str = "68c9ce2a167992d06678c4fdc0d1075b5dfff922",
                 log_system_interval: int = 10):
        """
        初始化WandB监控系统
        
        Args:
            project_name: WandB项目名称
            api_key: WandB API密钥
            log_system_interval: 系统监控记录间隔（批次数）
        """
        self.project_name = project_name
        self.api_key = api_key
        self.log_system_interval = log_system_interval
        self.batch_count = 0
        self.epoch_start_time = None
        self.batch_times = []
        
        # 登录WandB
        try:
            wandb.login(key=api_key)
            print("✅ WandB登录成功")
        except Exception as e:
            print(f"❌ WandB登录失败: {e}")
            raise
    
    def init_run(self, 
                 config: Dict[str, Any], 
                 run_name: Optional[str] = None,
                 tags: Optional[list] = None):
        """
        初始化WandB运行
        
        Args:
            config: 训练配置参数
            run_name: 运行名称
            tags: 标签列表
        """
        if tags is None:
            tags = ["optimization", "monitoring"]
        
        # 添加系统信息到配置
        config.update(self._get_system_info())
        
        wandb.init(
            project=self.project_name,
            config=config,
            name=run_name,
            tags=tags,
            reinit=True
        )
        
        print(f"🚀 WandB运行初始化完成: {wandb.run.name}")
        
        # 记录初始系统状态
        self.log_system_metrics(force=True)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统基础信息"""
        info = {
            'system/python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            'system/cpu_count': psutil.cpu_count(),
            'system/memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'system/torch_version': torch.__version__,
            'system/cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            # 获取详细GPU信息
            gpu_props = torch.cuda.get_device_properties(0)
            capability = torch.cuda.get_device_capability(0)

            info.update({
                'system/cuda_version': torch.version.cuda,
                'system/gpu_count': torch.cuda.device_count(),
                'system/gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown",
                'system/gpu_memory_total_gb': gpu_props.total_memory / (1024**3),
                'system/gpu_compute_capability': f"{capability[0]}.{capability[1]}",
                'system/gpu_multiprocessor_count': gpu_props.multi_processor_count,
            })
        
        return info
    
    def log_system_metrics(self, force: bool = False):
        """
        记录系统性能指标
        
        Args:
            force: 强制记录（忽略间隔限制）
        """
        if not force and self.batch_count % self.log_system_interval != 0:
            return
        
        metrics = {}
        
        # CPU和内存监控
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        metrics.update({
            'system/cpu_percent': cpu_percent,
            'system/memory_percent': memory.percent,
            'system/memory_used_gb': memory.used / (1024**3),
            'system/memory_available_gb': memory.available / (1024**3),
        })
        
        # GPU监控
        if GPU_AVAILABLE:
            gpu_metrics = self._get_gpu_metrics()
            metrics.update(gpu_metrics)
        
        # 详细GPU监控（如果可用）
        if NVML_AVAILABLE:
            detailed_gpu_metrics = self._get_detailed_gpu_metrics()
            metrics.update(detailed_gpu_metrics)
        
        # 进程监控
        process_metrics = self._get_process_metrics()
        metrics.update(process_metrics)
        
        wandb.log(metrics)
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """获取GPU基础监控指标"""
        metrics = {}
        
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                prefix = f'gpu_{i}'
                metrics.update({
                    f'{prefix}/utilization_percent': gpu.load * 100,
                    f'{prefix}/memory_used_mb': gpu.memoryUsed,
                    f'{prefix}/memory_total_mb': gpu.memoryTotal,
                    f'{prefix}/memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    f'{prefix}/memory_free_mb': gpu.memoryFree,
                    f'{prefix}/temperature_c': gpu.temperature,
                })
        except Exception as e:
            print(f"⚠️ GPU监控错误: {e}")
        
        return metrics
    
    def _get_detailed_gpu_metrics(self) -> Dict[str, float]:
        """获取详细GPU监控指标（需要nvidia-ml-py3）"""
        metrics = {}
        
        try:
            device_count = nvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                prefix = f'gpu_{i}'
                
                # 功耗监控
                try:
                    power_draw = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                    power_limit = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                    metrics[f'{prefix}/power_draw_w'] = power_draw
                    metrics[f'{prefix}/power_limit_w'] = power_limit
                    metrics[f'{prefix}/power_percent'] = (power_draw / power_limit) * 100
                except:
                    pass
                
                # 时钟频率
                try:
                    graphics_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                    metrics[f'{prefix}/graphics_clock_mhz'] = graphics_clock
                    metrics[f'{prefix}/memory_clock_mhz'] = memory_clock
                except:
                    pass
                
                # 风扇速度
                try:
                    fan_speed = nvml.nvmlDeviceGetFanSpeed(handle)
                    metrics[f'{prefix}/fan_speed_percent'] = fan_speed
                except:
                    pass
        
        except Exception as e:
            print(f"⚠️ 详细GPU监控错误: {e}")
        
        return metrics
    
    def _get_process_metrics(self) -> Dict[str, float]:
        """获取当前进程监控指标"""
        metrics = {}
        
        try:
            process = psutil.Process()
            
            # 进程CPU和内存
            metrics.update({
                'process/cpu_percent': process.cpu_percent(),
                'process/memory_percent': process.memory_percent(),
                'process/memory_rss_gb': process.memory_info().rss / (1024**3),
                'process/memory_vms_gb': process.memory_info().vms / (1024**3),
                'process/num_threads': process.num_threads(),
            })
            
            # 文件描述符
            try:
                metrics['process/num_fds'] = process.num_fds()
            except:
                pass  # Windows不支持
                
        except Exception as e:
            print(f"⚠️ 进程监控错误: {e}")
        
        return metrics
    
    def log_training_metrics(self, 
                           epoch: int,
                           batch_idx: int,
                           loss: float,
                           learning_rate: float,
                           **kwargs):
        """
        记录训练指标
        
        Args:
            epoch: 当前epoch
            batch_idx: 当前batch索引
            loss: 损失值
            learning_rate: 学习率
            **kwargs: 其他训练指标
        """
        self.batch_count += 1
        
        # 计算训练速度
        current_time = time.time()
        if hasattr(self, '_last_batch_time'):
            batch_time = current_time - self._last_batch_time
            self.batch_times.append(batch_time)
            
            # 保持最近100个batch的时间记录
            if len(self.batch_times) > 100:
                self.batch_times.pop(0)
            
            avg_batch_time = np.mean(self.batch_times)
            batches_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
        else:
            batches_per_sec = 0
        
        self._last_batch_time = current_time
        
        # 基础训练指标
        metrics = {
            'train/epoch': epoch,
            'train/batch': batch_idx,
            'train/loss': loss,
            'train/learning_rate': learning_rate,
            'train/batches_per_sec': batches_per_sec,
            'train/global_step': self.batch_count,
        }
        
        # 添加其他指标
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                metrics[f'train/{key}'] = value
        
        wandb.log(metrics)
        
        # 定期记录系统指标
        self.log_system_metrics()
    
    def log_validation_metrics(self, epoch: int, metrics_dict: Dict[str, float]):
        """记录验证指标"""
        val_metrics = {}
        for key, value in metrics_dict.items():
            val_metrics[f'val/{key}'] = value
        
        val_metrics['val/epoch'] = epoch
        wandb.log(val_metrics)
    
    def log_test_metrics(self, epoch: int, metrics_dict: Dict[str, float]):
        """记录测试指标"""
        test_metrics = {}
        for key, value in metrics_dict.items():
            test_metrics[f'test/{key}'] = value
        
        test_metrics['test/epoch'] = epoch
        wandb.log(test_metrics)
    
    def log_model_info(self, model):
        """记录模型信息"""
        try:
            # 计算参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                'model/total_parameters': total_params,
                'model/trainable_parameters': trainable_params,
                'model/non_trainable_parameters': total_params - trainable_params,
                'model/size_mb': total_params * 4 / (1024**2),  # 假设float32
            }
            
            wandb.log(model_info)
            
            # 记录模型结构
            wandb.watch(model, log="all", log_freq=1000)
            
        except Exception as e:
            print(f"⚠️ 模型信息记录错误: {e}")
    
    def start_epoch(self, epoch: int):
        """开始新的epoch"""
        self.epoch_start_time = time.time()
        wandb.log({'epoch/start': epoch})
    
    def end_epoch(self, epoch: int):
        """结束当前epoch"""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            wandb.log({
                'epoch/duration_seconds': epoch_time,
                'epoch/duration_minutes': epoch_time / 60,
                'epoch/end': epoch
            })
    
    def log_memory_usage(self, stage: str = ""):
        """记录详细内存使用情况"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
                
                prefix = f'memory/gpu_{i}'
                if stage:
                    prefix += f'/{stage}'
                
                wandb.log({
                    f'{prefix}/allocated_gb': allocated,
                    f'{prefix}/reserved_gb': reserved,
                    f'{prefix}/max_allocated_gb': max_allocated,
                })
    
    def finish(self):
        """结束WandB运行"""
        try:
            wandb.finish()
            print("✅ WandB运行已结束")
        except Exception as e:
            print(f"⚠️ WandB结束时出错: {e}")


def install_dependencies():
    """安装必要的依赖包"""
    try:
        import GPUtil
        import nvidia_ml_py3
        print("✅ 所有依赖已安装")
    except ImportError as e:
        print(f"⚠️ 缺少依赖: {e}")
        print("请运行以下命令安装:")
        print("pip install GPUtil nvidia-ml-py3 wandb")


if __name__ == "__main__":
    # 测试WandB监控系统
    install_dependencies()
    
    logger = WandBLogger()
    
    # 测试配置
    test_config = {
        'batch_size': 4,
        'learning_rate': 0.001,
        'epochs': 2,
        'model': 'R2Gen',
        'dataset': 'iu_xray'
    }
    
    logger.init_run(test_config, run_name="wandb_test")
    
    # 模拟训练过程
    for epoch in range(2):
        logger.start_epoch(epoch)
        
        for batch in range(10):
            # 模拟训练指标
            loss = 3.0 - epoch * 0.5 - batch * 0.01
            lr = 0.001 * (0.9 ** epoch)
            
            logger.log_training_metrics(
                epoch=epoch,
                batch_idx=batch,
                loss=loss,
                learning_rate=lr,
                gradient_norm=0.5
            )
            
            time.sleep(0.1)  # 模拟训练时间
        
        # 模拟验证
        val_metrics = {
            'loss': 2.5 - epoch * 0.3,
            'BLEU_4': 0.1 + epoch * 0.05,
            'METEOR': 0.15 + epoch * 0.03
        }
        logger.log_validation_metrics(epoch, val_metrics)
        
        logger.end_epoch(epoch)
    
    logger.finish()
    print("🎉 WandB测试完成！")

"""
显存监控模块 - 实时监控GPU显存使用情况
针对8GB RTX 4070 Laptop GPU优化
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False
    warnings.warn("GPUtil not available, some GPU monitoring features will be disabled")


class MemoryMonitor:
    """
    GPU显存监控器
    
    功能：
    - 实时监控显存使用情况
    - 检测显存使用趋势
    - 提供安全性检查
    - 记录历史数据用于分析
    """
    
    def __init__(self, 
                 target_utilization: float = 0.85,
                 device_id: int = 0,
                 history_size: int = 100):
        """
        初始化显存监控器
        
        Args:
            target_utilization: 目标显存利用率 (0.85 = 85%)
            device_id: GPU设备ID
            history_size: 历史记录保存数量
        """
        self.device_id = device_id
        self.target_utilization = target_utilization
        self.history_size = history_size
        
        # 获取GPU基础信息
        if torch.cuda.is_available():
            self.device_props = torch.cuda.get_device_properties(device_id)
            self.total_memory = self.device_props.total_memory
            self.total_memory_gb = self.total_memory / (1024**3)
            self.safe_memory = self.total_memory * target_utilization
            self.safe_memory_gb = self.safe_memory / (1024**3)
        else:
            raise RuntimeError("CUDA not available")
        
        # 历史记录
        self.memory_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        
        # 统计信息
        self.peak_memory_allocated = 0
        self.peak_memory_reserved = 0
        self.oom_count = 0
        self.last_check_time = time.time()
        
        print(f"✅ 显存监控器初始化完成")
        print(f"   GPU: {self.device_props.name}")
        print(f"   总显存: {self.total_memory_gb:.2f} GB")
        print(f"   目标使用率: {target_utilization*100:.1f}%")
        print(f"   安全阈值: {self.safe_memory_gb:.2f} GB")
    
    def get_current_usage(self) -> Dict[str, float]:
        """
        获取当前显存使用情况
        
        Returns:
            包含详细显存信息的字典
        """
        current_time = time.time()
        
        # PyTorch显存信息
        allocated = torch.cuda.memory_allocated(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        free_pytorch = self.total_memory - reserved
        
        # 计算使用率
        allocated_percent = (allocated / self.total_memory) * 100
        reserved_percent = (reserved / self.total_memory) * 100
        free_percent = (free_pytorch / self.total_memory) * 100
        
        usage_info = {
            # 绝对值 (字节)
            'allocated_bytes': allocated,
            'reserved_bytes': reserved,
            'free_bytes': free_pytorch,
            'total_bytes': self.total_memory,
            
            # GB单位
            'allocated_gb': allocated / (1024**3),
            'reserved_gb': reserved / (1024**3),
            'free_gb': free_pytorch / (1024**3),
            'total_gb': self.total_memory_gb,
            
            # 百分比
            'allocated_percent': allocated_percent,
            'reserved_percent': reserved_percent,
            'free_percent': free_percent,
            'utilization_percent': reserved_percent,  # 主要使用率指标
            
            # 时间戳
            'timestamp': current_time,
            
            # 安全性指标
            'is_safe': reserved < self.safe_memory,
            'safety_margin_gb': (self.safe_memory - reserved) / (1024**3),
            'safety_margin_percent': ((self.safe_memory - reserved) / self.total_memory) * 100
        }
        
        # 更新峰值记录
        if allocated > self.peak_memory_allocated:
            self.peak_memory_allocated = allocated
        if reserved > self.peak_memory_reserved:
            self.peak_memory_reserved = reserved
        
        # 添加到历史记录
        self.memory_history.append(usage_info.copy())
        self.timestamp_history.append(current_time)
        
        # 更新检查时间
        self.last_check_time = current_time
        
        return usage_info
    
    def get_gpu_util_info(self) -> Optional[Dict[str, float]]:
        """
        使用GPUtil获取额外的GPU信息
        
        Returns:
            GPU利用率、温度等信息，如果GPUtil不可用则返回None
        """
        if not GPU_UTIL_AVAILABLE:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus and len(gpus) > self.device_id:
                gpu = gpus[self.device_id]
                return {
                    'gpu_utilization_percent': gpu.load * 100,
                    'gpu_memory_used_mb': gpu.memoryUsed,
                    'gpu_memory_total_mb': gpu.memoryTotal,
                    'gpu_memory_free_mb': gpu.memoryFree,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_temperature_c': gpu.temperature,
                    'gpu_name': gpu.name,
                    'gpu_uuid': gpu.uuid
                }
        except Exception as e:
            warnings.warn(f"Failed to get GPU util info: {e}")
        
        return None
    
    def get_comprehensive_status(self) -> Dict[str, any]:
        """
        获取综合的显存和GPU状态信息
        
        Returns:
            包含PyTorch和GPUtil信息的综合状态
        """
        status = {
            'memory': self.get_current_usage(),
            'gpu_util': self.get_gpu_util_info(),
            'statistics': self.get_statistics(),
            'trends': self.get_trends()
        }
        
        return status
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取统计信息
        
        Returns:
            峰值使用、平均使用等统计数据
        """
        stats = {
            'peak_allocated_gb': self.peak_memory_allocated / (1024**3),
            'peak_reserved_gb': self.peak_memory_reserved / (1024**3),
            'peak_allocated_percent': (self.peak_memory_allocated / self.total_memory) * 100,
            'peak_reserved_percent': (self.peak_memory_reserved / self.total_memory) * 100,
            'oom_count': self.oom_count,
            'monitoring_duration_minutes': (time.time() - self.timestamp_history[0]) / 60 if self.timestamp_history else 0
        }
        
        # 计算历史平均值
        if len(self.memory_history) > 0:
            recent_usage = [entry['utilization_percent'] for entry in self.memory_history]
            stats.update({
                'avg_utilization_percent': np.mean(recent_usage),
                'max_utilization_percent': np.max(recent_usage),
                'min_utilization_percent': np.min(recent_usage),
                'std_utilization_percent': np.std(recent_usage)
            })
        
        return stats
    
    def get_trends(self) -> Dict[str, any]:
        """
        分析显存使用趋势
        
        Returns:
            趋势分析结果
        """
        if len(self.memory_history) < 10:
            return {'trend': 'insufficient_data', 'slope': 0, 'prediction': None}
        
        # 获取最近的使用率数据
        recent_usage = [entry['utilization_percent'] for entry in list(self.memory_history)[-20:]]
        recent_times = list(range(len(recent_usage)))
        
        # 计算趋势斜率
        slope = np.polyfit(recent_times, recent_usage, 1)[0]
        
        # 判断趋势
        if abs(slope) < 0.1:
            trend = 'stable'
        elif slope > 0.1:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        # 预测未来使用率（简单线性预测）
        if len(recent_usage) > 0:
            current_usage = recent_usage[-1]
            predicted_usage_10_steps = current_usage + slope * 10
            predicted_usage_10_steps = max(0, min(100, predicted_usage_10_steps))
        else:
            predicted_usage_10_steps = None
        
        return {
            'trend': trend,
            'slope': slope,
            'current_usage': recent_usage[-1] if recent_usage else 0,
            'predicted_usage_10_steps': predicted_usage_10_steps,
            'is_memory_leak_suspected': slope > 1.0,  # 每步增长超过1%可能是内存泄漏
            'samples_count': len(recent_usage)
        }
    
    def is_safe_to_increase(self, estimated_increase_gb: float) -> Tuple[bool, Dict[str, any]]:
        """
        判断是否可以安全增加显存使用
        
        Args:
            estimated_increase_gb: 预计增加的显存使用量(GB)
            
        Returns:
            (是否安全, 详细信息)
        """
        current = self.get_current_usage()
        projected_usage_gb = current['reserved_gb'] + estimated_increase_gb
        projected_percent = (projected_usage_gb / self.total_memory_gb) * 100
        
        is_safe = projected_usage_gb <= self.safe_memory_gb
        
        safety_info = {
            'current_usage_gb': current['reserved_gb'],
            'estimated_increase_gb': estimated_increase_gb,
            'projected_usage_gb': projected_usage_gb,
            'projected_percent': projected_percent,
            'target_percent': self.target_utilization * 100,
            'is_safe': is_safe,
            'margin_gb': self.safe_memory_gb - projected_usage_gb,
            'margin_percent': ((self.safe_memory_gb - projected_usage_gb) / self.total_memory_gb) * 100
        }
        
        return is_safe, safety_info
    
    def record_oom_event(self, context: str = ""):
        """
        记录OOM事件
        
        Args:
            context: OOM发生的上下文信息
        """
        self.oom_count += 1
        oom_info = {
            'timestamp': time.time(),
            'context': context,
            'memory_state': self.get_current_usage(),
            'oom_count': self.oom_count
        }
        
        print(f"🚨 OOM事件记录 #{self.oom_count}")
        print(f"   时间: {time.strftime('%H:%M:%S')}")
        print(f"   上下文: {context}")
        print(f"   显存使用: {oom_info['memory_state']['utilization_percent']:.1f}%")
        
        return oom_info
    
    def clear_cache_and_reset(self):
        """
        清理显存缓存并重置统计
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device_id)
        
        # 重置峰值统计
        self.peak_memory_allocated = torch.cuda.memory_allocated(self.device_id)
        self.peak_memory_reserved = torch.cuda.memory_reserved(self.device_id)
        
        print("🧹 显存缓存已清理，统计已重置")
    
    def print_status(self, detailed: bool = False):
        """
        打印当前状态
        
        Args:
            detailed: 是否显示详细信息
        """
        status = self.get_comprehensive_status()
        memory = status['memory']
        gpu_util = status['gpu_util']
        stats = status['statistics']
        trends = status['trends']
        
        print(f"\n📊 显存监控状态 ({time.strftime('%H:%M:%S')})")
        print(f"   显存使用: {memory['utilization_percent']:.1f}% ({memory['reserved_gb']:.2f}GB / {memory['total_gb']:.2f}GB)")
        print(f"   安全状态: {'✅ 安全' if memory['is_safe'] else '⚠️ 接近上限'}")
        print(f"   安全余量: {memory['safety_margin_gb']:.2f}GB ({memory['safety_margin_percent']:.1f}%)")
        
        if gpu_util:
            print(f"   GPU利用率: {gpu_util['gpu_utilization_percent']:.1f}%")
            print(f"   GPU温度: {gpu_util['gpu_temperature_c']}°C")
        
        if detailed:
            print(f"\n📈 统计信息:")
            print(f"   峰值使用: {stats['peak_reserved_percent']:.1f}%")
            print(f"   平均使用: {stats.get('avg_utilization_percent', 0):.1f}%")
            print(f"   使用趋势: {trends['trend']}")
            if trends.get('is_memory_leak_suspected', False):
                print(f"   ⚠️ 疑似内存泄漏 (斜率: {trends['slope']:.2f})")


if __name__ == "__main__":
    # 测试显存监控器
    print("🧪 测试显存监控器...")
    
    monitor = MemoryMonitor()
    
    # 基础状态检查
    monitor.print_status(detailed=True)
    
    # 模拟一些显存使用
    print("\n🔬 模拟显存使用测试...")
    test_tensors = []
    
    for i in range(5):
        # 创建一些张量来使用显存
        tensor = torch.randn(1000, 1000, device='cuda')
        test_tensors.append(tensor)
        
        # 检查状态
        status = monitor.get_current_usage()
        print(f"   步骤 {i+1}: {status['utilization_percent']:.1f}% 显存使用")
        
        # 测试安全性检查
        is_safe, safety_info = monitor.is_safe_to_increase(0.5)  # 测试增加500MB
        print(f"   增加500MB安全性: {'✅' if is_safe else '❌'}")
    
    # 清理测试张量
    del test_tensors
    monitor.clear_cache_and_reset()
    
    # 最终状态
    print("\n📊 清理后状态:")
    monitor.print_status()
    
    print("✅ 显存监控器测试完成！")

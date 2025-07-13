"""
动态Batch Size优化器
自动找到最适合8GB显存的最优batch size配置
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

try:
    from .memory_monitor import MemoryMonitor
except ImportError:
    from memory_monitor import MemoryMonitor


class BatchSizeOptimizer:
    """
    动态Batch Size优化器
    
    功能：
    - 自动测试不同batch size的显存使用
    - 找到最优batch size配置
    - 支持不同精度模式 (FP32, FP16, FP8)
    - OOM保护和恢复机制
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 memory_monitor: MemoryMonitor,
                 device_id: int = 0):
        """
        初始化Batch Size优化器
        
        Args:
            model: 要优化的模型
            memory_monitor: 显存监控器
            device_id: GPU设备ID
        """
        self.model = model
        self.memory_monitor = memory_monitor
        self.device_id = device_id
        self.device = f'cuda:{device_id}'
        
        # 测试结果记录
        self.test_results = []
        self.optimization_history = []
        
        # 安全配置
        self.safety_margin_gb = 0.5  # 500MB安全余量
        self.max_test_batch_size = 64  # 最大测试batch size
        
        print(f"✅ Batch Size优化器初始化完成")
        print(f"   目标显存利用率: {memory_monitor.target_utilization*100:.1f}%")
        print(f"   安全余量: {self.safety_margin_gb:.1f}GB")
    
    def create_sample_batch(self, 
                           batch_size: int,
                           image_size: Tuple[int, int] = (224, 224),
                           sequence_length: int = 60) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建用于测试的样本batch
        
        Args:
            batch_size: batch大小
            image_size: 图像尺寸
            sequence_length: 序列长度
            
        Returns:
            (images, reports_ids, reports_masks)
        """
        # 创建模拟的输入数据
        images = torch.randn(batch_size, 3, *image_size, device=self.device)
        reports_ids = torch.randint(0, 1000, (batch_size, sequence_length), device=self.device)
        reports_masks = torch.ones(batch_size, sequence_length, device=self.device)
        
        return images, reports_ids, reports_masks
    
    def test_batch_size(self, 
                       batch_size: int,
                       precision: str = 'fp32',
                       num_test_steps: int = 3) -> Dict[str, Any]:
        """
        测试特定batch size的显存使用和性能
        
        Args:
            batch_size: 要测试的batch size
            precision: 精度模式 ('fp32', 'fp16', 'fp8')
            num_test_steps: 测试步数
            
        Returns:
            测试结果字典
        """
        print(f"  🔬 测试 batch_size={batch_size}, precision={precision}")
        
        # 清理显存
        torch.cuda.empty_cache()
        self.memory_monitor.clear_cache_and_reset()
        
        # 记录初始状态
        initial_memory = self.memory_monitor.get_current_usage()
        
        try:
            # 创建测试数据
            test_data = self.create_sample_batch(batch_size)
            
            # 设置模型为训练模式
            self.model.train()
            
            # 记录测试开始时间
            start_time = time.time()
            step_times = []
            memory_snapshots = []
            
            # 执行测试步骤
            for step in range(num_test_steps):
                step_start = time.time()
                
                # 前向传播
                if precision == 'fp16':
                    with torch.amp.autocast('cuda'):
                        output = self.model(test_data[0], test_data[1], mode='train')
                        # 模拟损失计算
                        loss = output.mean()
                elif precision == 'fp8':
                    # FP8支持需要特殊处理，这里先用FP16代替
                    with torch.amp.autocast('cuda'):
                        output = self.model(test_data[0], test_data[1], mode='train')
                        loss = output.mean()
                else:  # fp32
                    output = self.model(test_data[0], test_data[1], mode='train')
                    loss = output.mean()
                
                # 反向传播
                loss.backward()
                
                # 记录显存使用
                memory_snapshot = self.memory_monitor.get_current_usage()
                memory_snapshots.append(memory_snapshot)
                
                # 清理梯度
                self.model.zero_grad()
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # 检查是否超出安全范围
                if not memory_snapshot['is_safe']:
                    print(f"    ⚠️ 超出安全范围: {memory_snapshot['utilization_percent']:.1f}%")
                    break
            
            # 计算统计信息
            total_time = time.time() - start_time
            avg_step_time = np.mean(step_times)
            peak_memory = max(snapshot['utilization_percent'] for snapshot in memory_snapshots)
            avg_memory = np.mean([snapshot['utilization_percent'] for snapshot in memory_snapshots])
            
            # 计算性能指标
            samples_per_second = batch_size / avg_step_time if avg_step_time > 0 else 0
            batches_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0
            
            # 清理测试数据
            del test_data, output, loss
            torch.cuda.empty_cache()
            
            # 最终显存状态
            final_memory = self.memory_monitor.get_current_usage()
            
            result = {
                'batch_size': batch_size,
                'precision': precision,
                'success': True,
                'peak_memory_percent': peak_memory,
                'avg_memory_percent': avg_memory,
                'peak_memory_gb': peak_memory * self.memory_monitor.total_memory_gb / 100,
                'avg_step_time_seconds': avg_step_time,
                'samples_per_second': samples_per_second,
                'batches_per_second': batches_per_second,
                'total_test_time': total_time,
                'num_steps_completed': len(step_times),
                'is_safe': all(snapshot['is_safe'] for snapshot in memory_snapshots),
                'memory_efficiency': (batch_size * len(step_times)) / peak_memory if peak_memory > 0 else 0,
                'initial_memory_gb': initial_memory['reserved_gb'],
                'final_memory_gb': final_memory['reserved_gb'],
                'memory_increase_gb': final_memory['reserved_gb'] - initial_memory['reserved_gb']
            }
            
            print(f"    ✅ 成功: {peak_memory:.1f}% 峰值显存, {samples_per_second:.1f} samples/sec")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    ❌ OOM: batch_size={batch_size} 超出显存限制")
                
                # 记录OOM事件
                self.memory_monitor.record_oom_event(f"batch_size={batch_size}, precision={precision}")
                
                result = {
                    'batch_size': batch_size,
                    'precision': precision,
                    'success': False,
                    'error': 'OOM',
                    'error_message': str(e),
                    'peak_memory_percent': 100.0,  # 假设达到了100%
                    'is_safe': False
                }
                
                # 清理显存
                torch.cuda.empty_cache()
            else:
                print(f"    ❌ 其他错误: {e}")
                result = {
                    'batch_size': batch_size,
                    'precision': precision,
                    'success': False,
                    'error': 'other',
                    'error_message': str(e),
                    'is_safe': False
                }
        
        return result
    
    def find_optimal_batch_size(self,
                               start_size: int = 4,
                               max_size: int = 32,
                               precision: str = 'fp32',
                               step_size: int = 2) -> Tuple[int, List[Dict]]:
        """
        找到最优的batch size
        
        Args:
            start_size: 起始batch size
            max_size: 最大batch size
            precision: 精度模式
            step_size: 步长
            
        Returns:
            (最优batch_size, 测试结果列表)
        """
        print(f"\n🔍 开始{precision}精度的batch size优化...")
        print(f"   测试范围: {start_size} - {max_size}, 步长: {step_size}")
        
        test_results = []
        optimal_batch_size = start_size
        best_efficiency = 0
        
        # 逐步测试不同的batch size
        current_size = start_size
        while current_size <= max_size:
            result = self.test_batch_size(current_size, precision)
            test_results.append(result)
            
            if result['success'] and result['is_safe']:
                # 计算效率分数 (考虑batch size和显存利用率)
                efficiency = result.get('memory_efficiency', 0)
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    optimal_batch_size = current_size
                
                # 如果显存使用率超过目标，停止测试
                if result['peak_memory_percent'] > self.memory_monitor.target_utilization * 100:
                    print(f"  ⚠️ 达到目标显存使用率，停止测试")
                    break
            else:
                # 如果失败，停止测试
                print(f"  ❌ batch_size={current_size} 失败，停止测试")
                break
            
            current_size += step_size
        
        # 保存测试结果
        self.test_results.extend(test_results)
        
        # 记录优化历史
        optimization_record = {
            'timestamp': time.time(),
            'precision': precision,
            'optimal_batch_size': optimal_batch_size,
            'best_efficiency': best_efficiency,
            'test_range': (start_size, min(current_size - step_size, max_size)),
            'num_tests': len(test_results),
            'successful_tests': len([r for r in test_results if r['success']])
        }
        self.optimization_history.append(optimization_record)
        
        print(f"\n✅ {precision}优化完成!")
        print(f"   最优batch size: {optimal_batch_size}")
        print(f"   最佳效率分数: {best_efficiency:.2f}")
        print(f"   成功测试: {optimization_record['successful_tests']}/{optimization_record['num_tests']}")
        
        return optimal_batch_size, test_results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        获取优化结果摘要
        
        Returns:
            优化结果摘要
        """
        if not self.test_results:
            return {'status': 'no_tests_run'}
        
        successful_tests = [r for r in self.test_results if r['success']]
        
        if not successful_tests:
            return {'status': 'all_tests_failed'}
        
        # 找到最优配置
        best_test = max(successful_tests, key=lambda x: x.get('memory_efficiency', 0))
        
        summary = {
            'status': 'success',
            'optimal_batch_size': best_test['batch_size'],
            'optimal_precision': best_test['precision'],
            'peak_memory_percent': best_test['peak_memory_percent'],
            'peak_memory_gb': best_test['peak_memory_gb'],
            'samples_per_second': best_test['samples_per_second'],
            'memory_efficiency': best_test['memory_efficiency'],
            'total_tests': len(self.test_results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(self.test_results) - len(successful_tests),
            'oom_count': self.memory_monitor.oom_count,
            'optimization_history': self.optimization_history
        }
        
        return summary
    
    def print_optimization_report(self):
        """
        打印详细的优化报告
        """
        summary = self.get_optimization_summary()
        
        if summary['status'] != 'success':
            print(f"❌ 优化失败: {summary['status']}")
            return
        
        print(f"\n📊 Batch Size优化报告")
        print(f"=" * 50)
        print(f"✅ 最优配置:")
        print(f"   Batch Size: {summary['optimal_batch_size']}")
        print(f"   精度模式: {summary['optimal_precision']}")
        print(f"   峰值显存: {summary['peak_memory_percent']:.1f}% ({summary['peak_memory_gb']:.2f}GB)")
        print(f"   训练速度: {summary['samples_per_second']:.1f} samples/sec")
        print(f"   内存效率: {summary['memory_efficiency']:.2f}")
        
        print(f"\n📈 测试统计:")
        print(f"   总测试数: {summary['total_tests']}")
        print(f"   成功测试: {summary['successful_tests']}")
        print(f"   失败测试: {summary['failed_tests']}")
        print(f"   OOM次数: {summary['oom_count']}")
        
        # 显示所有成功的测试结果
        successful_tests = [r for r in self.test_results if r['success']]
        if successful_tests:
            print(f"\n📋 详细测试结果:")
            print(f"{'Batch Size':<10} {'Memory %':<10} {'Speed':<15} {'Efficiency':<12}")
            print(f"{'-'*50}")
            for test in successful_tests:
                print(f"{test['batch_size']:<10} "
                      f"{test['peak_memory_percent']:<10.1f} "
                      f"{test['samples_per_second']:<15.1f} "
                      f"{test.get('memory_efficiency', 0):<12.2f}")


if __name__ == "__main__":
    # 测试Batch Size优化器
    print("🧪 测试Batch Size优化器...")
    
    # 这里需要一个简单的测试模型
    class SimpleTestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((7, 7))
            self.fc = torch.nn.Linear(64 * 7 * 7, 512)
            self.output = torch.nn.Linear(512, 1000)
        
        def forward(self, images, reports_ids=None, mode='train'):
            x = self.conv(images)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.output(x)
            return x
    
    # 创建测试模型和监控器
    model = SimpleTestModel().cuda()
    memory_monitor = MemoryMonitor(target_utilization=0.85)
    
    # 创建优化器
    optimizer = BatchSizeOptimizer(model, memory_monitor)
    
    # 运行优化
    optimal_size, results = optimizer.find_optimal_batch_size(
        start_size=2,
        max_size=16,
        precision='fp32',
        step_size=2
    )
    
    # 打印报告
    optimizer.print_optimization_report()
    
    print("✅ Batch Size优化器测试完成！")

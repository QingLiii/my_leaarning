#!/usr/bin/env python3
"""
测试Batch Size优化器
"""

import sys
import torch
import time

# 添加模块路径
sys.path.append('R2Gen-main')
sys.path.append('R2Gen-main/modules')

from R2Gen_main.modules.memory_monitor import MemoryMonitor
from R2Gen_main.modules.batch_size_optimizer import BatchSizeOptimizer


class SimpleTestModel(torch.nn.Module):
    """简单的测试模型"""
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


def test_batch_optimizer():
    """测试Batch Size优化器"""
    print("🧪 测试Batch Size优化器...")
    
    # 创建测试模型和监控器
    model = SimpleTestModel().cuda()
    memory_monitor = MemoryMonitor(target_utilization=0.85)
    
    print(f"📊 初始显存状态:")
    memory_monitor.print_status()
    
    # 创建优化器
    optimizer = BatchSizeOptimizer(model, memory_monitor)
    
    # 运行优化
    print(f"\n🚀 开始优化...")
    optimal_size, results = optimizer.find_optimal_batch_size(
        start_size=2,
        max_size=16,
        precision='fp32',
        step_size=2
    )
    
    # 打印报告
    optimizer.print_optimization_report()
    
    print(f"\n📊 最终显存状态:")
    memory_monitor.print_status()
    
    return optimal_size, results


if __name__ == "__main__":
    try:
        optimal_size, results = test_batch_optimizer()
        print(f"\n✅ 测试完成! 最优batch size: {optimal_size}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""
测试显存优化模块
"""

import sys
import os
import torch
import time

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.memory_monitor import MemoryMonitor


class SimpleTestModel(torch.nn.Module):
    """简单的测试模型，模拟R2Gen的显存使用"""
    def __init__(self):
        super().__init__()
        # 模拟视觉编码器
        self.visual_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((14, 14))
        )
        
        # 模拟文本解码器
        self.text_decoder = torch.nn.Sequential(
            torch.nn.Linear(128 * 14 * 14, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1000),  # 词汇表大小
        )
        
        # 模拟注意力机制
        self.attention = torch.nn.MultiheadAttention(512, 8, batch_first=True)
    
    def forward(self, images, reports_ids=None, mode='train'):
        batch_size = images.size(0)
        
        # 视觉特征提取
        visual_features = self.visual_encoder(images)
        visual_features = visual_features.view(batch_size, -1)
        
        # 文本解码
        if mode == 'train' and reports_ids is not None:
            # 训练模式：使用teacher forcing
            seq_len = reports_ids.size(1)
            text_features = torch.randn(batch_size, seq_len, 512, device=images.device)
            
            # 注意力计算
            attended_features, _ = self.attention(text_features, text_features, text_features)
            
            # 输出预测
            output = self.text_decoder(visual_features.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, visual_features.size(1)))
            output = output.view(batch_size, seq_len, -1)
            
            return output
        else:
            # 推理模式
            return self.text_decoder(visual_features)


def test_memory_monitor():
    """测试显存监控器"""
    print("🧪 测试显存监控器...")
    
    monitor = MemoryMonitor(target_utilization=0.85)
    monitor.print_status(detailed=True)
    
    return monitor


def test_batch_size_optimization():
    """测试batch size优化"""
    print("\n🧪 测试Batch Size优化...")
    
    # 创建模型和监控器
    model = SimpleTestModel().cuda()
    monitor = MemoryMonitor(target_utilization=0.85)
    
    print(f"📊 模型加载后显存状态:")
    monitor.print_status()
    
    # 手动测试不同batch size
    test_sizes = [2, 4, 8, 12, 16, 20, 24]
    results = []
    
    print(f"\n🔬 开始batch size测试...")
    
    for batch_size in test_sizes:
        print(f"\n  测试 batch_size = {batch_size}")
        
        try:
            # 清理显存
            torch.cuda.empty_cache()
            monitor.clear_cache_and_reset()
            
            # 创建测试数据
            images = torch.randn(batch_size, 3, 224, 224, device='cuda')
            reports_ids = torch.randint(0, 1000, (batch_size, 60), device='cuda')
            
            # 记录开始状态
            start_memory = monitor.get_current_usage()
            
            # 前向传播
            start_time = time.time()
            model.train()
            
            with torch.cuda.amp.autocast():  # 使用FP16测试
                output = model(images, reports_ids, mode='train')
                loss = output.mean()
            
            # 反向传播
            loss.backward()
            
            # 记录结束状态
            end_time = time.time()
            end_memory = monitor.get_current_usage()
            
            # 计算性能指标
            step_time = end_time - start_time
            samples_per_sec = batch_size / step_time
            
            result = {
                'batch_size': batch_size,
                'success': True,
                'peak_memory_percent': end_memory['utilization_percent'],
                'peak_memory_gb': end_memory['reserved_gb'],
                'step_time': step_time,
                'samples_per_sec': samples_per_sec,
                'is_safe': end_memory['is_safe']
            }
            
            print(f"    ✅ 成功: {end_memory['utilization_percent']:.1f}% 显存, {samples_per_sec:.1f} samples/sec")
            
            # 清理
            del images, reports_ids, output, loss
            model.zero_grad()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    ❌ OOM: batch_size={batch_size}")
                result = {
                    'batch_size': batch_size,
                    'success': False,
                    'error': 'OOM',
                    'is_safe': False
                }
                
                # 清理显存
                torch.cuda.empty_cache()
            else:
                print(f"    ❌ 其他错误: {e}")
                result = {
                    'batch_size': batch_size,
                    'success': False,
                    'error': str(e),
                    'is_safe': False
                }
        
        results.append(result)
        
        # 如果失败，停止测试
        if not result['success']:
            break
    
    # 分析结果
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        # 找到最优配置
        # 优先考虑安全性，然后考虑显存利用率
        safe_results = [r for r in successful_results if r['is_safe']]
        
        if safe_results:
            # 在安全范围内找到显存利用率最高的
            optimal_result = max(safe_results, key=lambda x: x['peak_memory_percent'])
        else:
            # 如果没有安全的，选择显存使用最少的成功配置
            optimal_result = min(successful_results, key=lambda x: x['peak_memory_percent'])
        
        print(f"\n📊 优化结果摘要:")
        print(f"   测试范围: batch_size {test_sizes[0]} - {test_sizes[len(results)-1]}")
        print(f"   成功测试: {len(successful_results)}/{len(results)}")
        print(f"   最优配置: batch_size = {optimal_result['batch_size']}")
        print(f"   显存使用: {optimal_result['peak_memory_percent']:.1f}% ({optimal_result['peak_memory_gb']:.2f}GB)")
        print(f"   训练速度: {optimal_result['samples_per_sec']:.1f} samples/sec")
        print(f"   安全状态: {'✅ 安全' if optimal_result['is_safe'] else '⚠️ 接近上限'}")
        
        # 详细结果表格
        print(f"\n📋 详细测试结果:")
        print(f"{'Batch Size':<12} {'Memory %':<10} {'Memory GB':<12} {'Speed':<15} {'Status':<10}")
        print(f"{'-'*65}")
        
        for result in results:
            if result['success']:
                status = '✅ 安全' if result['is_safe'] else '⚠️ 接近'
                print(f"{result['batch_size']:<12} "
                      f"{result['peak_memory_percent']:<10.1f} "
                      f"{result['peak_memory_gb']:<12.2f} "
                      f"{result['samples_per_sec']:<15.1f} "
                      f"{status:<10}")
            else:
                print(f"{result['batch_size']:<12} "
                      f"{'N/A':<10} "
                      f"{'N/A':<12} "
                      f"{'N/A':<15} "
                      f"❌ {result['error']:<10}")
        
        return optimal_result
    else:
        print(f"❌ 所有测试都失败了")
        return None


def main():
    """主测试函数"""
    print("🚀 显存优化模块测试")
    print("=" * 50)
    
    # 测试1: 显存监控器
    monitor = test_memory_monitor()
    
    # 测试2: Batch size优化
    optimal_config = test_batch_size_optimization()
    
    # 最终状态
    print(f"\n📊 最终显存状态:")
    monitor.print_status()
    
    if optimal_config:
        print(f"\n🎯 推荐配置:")
        print(f"   batch_size = {optimal_config['batch_size']}")
        print(f"   预期显存使用: {optimal_config['peak_memory_percent']:.1f}%")
        print(f"   预期训练速度: {optimal_config['samples_per_sec']:.1f} samples/sec")
        
        # 计算相对于batch_size=4的提升
        baseline_speed = optimal_config['samples_per_sec'] * 4 / optimal_config['batch_size']  # 估算batch_size=4的速度
        speedup = optimal_config['samples_per_sec'] / baseline_speed
        
        print(f"   相对batch_size=4的提升: {speedup:.1f}倍")
    
    print(f"\n✅ 显存优化测试完成!")


if __name__ == "__main__":
    main()

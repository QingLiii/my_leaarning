"""
梯度累积管理器
智能管理梯度累积，保持等效大batch训练效果
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings


class GradientAccumulationManager:
    """
    梯度累积管理器
    
    功能：
    - 根据最优batch size计算累积步数
    - 保持等效大batch训练效果
    - 动态调整累积策略
    - 提供梯度同步控制
    """
    
    def __init__(self, 
                 target_effective_batch_size: int = 16,
                 min_accumulation_steps: int = 1,
                 max_accumulation_steps: int = 32):
        """
        初始化梯度累积管理器
        
        Args:
            target_effective_batch_size: 目标等效batch size
            min_accumulation_steps: 最小累积步数
            max_accumulation_steps: 最大累积步数
        """
        self.target_effective_batch_size = target_effective_batch_size
        self.min_accumulation_steps = min_accumulation_steps
        self.max_accumulation_steps = max_accumulation_steps
        
        # 当前配置
        self.current_batch_size = None
        self.accumulation_steps = 1
        self.effective_batch_size = None
        
        # 统计信息
        self.step_count = 0
        self.accumulation_count = 0
        self.gradient_updates = 0
        
        print(f"✅ 梯度累积管理器初始化完成")
        print(f"   目标等效batch size: {target_effective_batch_size}")
        print(f"   累积步数范围: {min_accumulation_steps} - {max_accumulation_steps}")
    
    def calculate_accumulation_steps(self, optimal_batch_size: int) -> int:
        """
        计算最优梯度累积步数
        
        Args:
            optimal_batch_size: 最优的实际batch size
            
        Returns:
            计算得到的累积步数
        """
        # 计算理想的累积步数
        ideal_steps = self.target_effective_batch_size / optimal_batch_size
        
        # 取整并限制在合理范围内
        accumulation_steps = max(
            self.min_accumulation_steps,
            min(self.max_accumulation_steps, round(ideal_steps))
        )
        
        # 更新配置
        self.current_batch_size = optimal_batch_size
        self.accumulation_steps = accumulation_steps
        self.effective_batch_size = optimal_batch_size * accumulation_steps
        
        # 重置统计
        self.step_count = 0
        self.accumulation_count = 0
        self.gradient_updates = 0
        
        print(f"\n📊 梯度累积配置计算完成:")
        print(f"   实际batch size: {optimal_batch_size}")
        print(f"   累积步数: {accumulation_steps}")
        print(f"   等效batch size: {self.effective_batch_size}")
        print(f"   目标batch size: {self.target_effective_batch_size}")
        
        # 计算效果评估
        efficiency = (self.effective_batch_size / self.target_effective_batch_size) * 100
        print(f"   目标达成率: {efficiency:.1f}%")
        
        if efficiency < 90:
            print(f"   ⚠️ 等效batch size低于目标，可能影响训练效果")
        elif efficiency > 110:
            print(f"   ⚠️ 等效batch size高于目标，可能影响收敛速度")
        else:
            print(f"   ✅ 等效batch size接近目标，配置合理")
        
        return accumulation_steps
    
    def should_update_gradients(self, step: int) -> bool:
        """
        判断是否应该更新梯度
        
        Args:
            step: 当前步数（从0开始）
            
        Returns:
            是否应该更新梯度
        """
        self.step_count += 1
        self.accumulation_count = (step + 1) % self.accumulation_steps
        
        should_update = (step + 1) % self.accumulation_steps == 0
        
        if should_update:
            self.gradient_updates += 1
        
        return should_update
    
    def get_loss_scale(self) -> float:
        """
        获取损失缩放因子
        
        Returns:
            损失缩放因子（用于梯度累积）
        """
        return 1.0 / self.accumulation_steps
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        获取当前配置信息
        
        Returns:
            当前配置字典
        """
        return {
            'current_batch_size': self.current_batch_size,
            'accumulation_steps': self.accumulation_steps,
            'effective_batch_size': self.effective_batch_size,
            'target_effective_batch_size': self.target_effective_batch_size,
            'loss_scale': self.get_loss_scale(),
            'efficiency_percent': (self.effective_batch_size / self.target_effective_batch_size) * 100 if self.effective_batch_size else 0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_steps': self.step_count,
            'gradient_updates': self.gradient_updates,
            'current_accumulation_count': self.accumulation_count,
            'steps_per_update': self.accumulation_steps,
            'avg_steps_per_update': self.step_count / self.gradient_updates if self.gradient_updates > 0 else 0,
            'update_efficiency': (self.gradient_updates * self.accumulation_steps) / self.step_count if self.step_count > 0 else 0
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.step_count = 0
        self.accumulation_count = 0
        self.gradient_updates = 0
        print("📊 梯度累积统计已重置")
    
    def update_target_batch_size(self, new_target: int):
        """
        更新目标等效batch size
        
        Args:
            new_target: 新的目标等效batch size
        """
        old_target = self.target_effective_batch_size
        self.target_effective_batch_size = new_target
        
        # 如果已经有配置，重新计算
        if self.current_batch_size is not None:
            self.calculate_accumulation_steps(self.current_batch_size)
        
        print(f"🔄 目标等效batch size已更新: {old_target} → {new_target}")
    
    def optimize_for_memory_constraint(self, 
                                     available_batch_sizes: List[int],
                                     memory_constraints: List[float]) -> Tuple[int, int]:
        """
        在内存约束下优化配置
        
        Args:
            available_batch_sizes: 可用的batch size列表
            memory_constraints: 对应的内存使用率列表
            
        Returns:
            (最优batch_size, 对应的累积步数)
        """
        best_config = None
        best_score = -1
        
        for batch_size, memory_usage in zip(available_batch_sizes, memory_constraints):
            # 计算这个batch size的累积步数
            accumulation_steps = max(
                self.min_accumulation_steps,
                min(self.max_accumulation_steps, round(self.target_effective_batch_size / batch_size))
            )
            
            effective_batch = batch_size * accumulation_steps
            
            # 计算评分（考虑目标达成率和内存效率）
            target_efficiency = min(1.0, effective_batch / self.target_effective_batch_size)
            memory_efficiency = 1.0 - (memory_usage / 100.0)  # 内存使用越少越好
            
            # 综合评分
            score = target_efficiency * 0.7 + memory_efficiency * 0.3
            
            if score > best_score:
                best_score = score
                best_config = (batch_size, accumulation_steps, effective_batch, memory_usage)
        
        if best_config:
            batch_size, accumulation_steps, effective_batch, memory_usage = best_config
            
            print(f"\n🎯 内存约束下的最优配置:")
            print(f"   batch size: {batch_size}")
            print(f"   累积步数: {accumulation_steps}")
            print(f"   等效batch: {effective_batch}")
            print(f"   内存使用: {memory_usage:.1f}%")
            print(f"   综合评分: {best_score:.3f}")
            
            return batch_size, accumulation_steps
        else:
            print(f"❌ 无法找到合适的配置")
            return available_batch_sizes[0], self.min_accumulation_steps
    
    def print_status(self, detailed: bool = False):
        """
        打印当前状态
        
        Args:
            detailed: 是否显示详细信息
        """
        config = self.get_current_config()
        stats = self.get_statistics()
        
        print(f"\n📊 梯度累积状态:")
        print(f"   当前配置: batch_size={config['current_batch_size']}, "
              f"accumulation={config['accumulation_steps']}, "
              f"effective={config['effective_batch_size']}")
        print(f"   目标达成: {config['efficiency_percent']:.1f}%")
        print(f"   损失缩放: {config['loss_scale']:.3f}")
        
        if detailed and stats['total_steps'] > 0:
            print(f"\n📈 统计信息:")
            print(f"   总步数: {stats['total_steps']}")
            print(f"   梯度更新: {stats['gradient_updates']}")
            print(f"   当前累积: {stats['current_accumulation_count']}/{config['accumulation_steps']}")
            print(f"   更新效率: {stats['update_efficiency']:.1%}")


class AdaptiveGradientAccumulation(GradientAccumulationManager):
    """
    自适应梯度累积管理器
    可以根据训练过程动态调整累积策略
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 自适应参数
        self.gradient_norm_history = []
        self.loss_history = []
        self.adaptation_enabled = True
        self.adaptation_interval = 100  # 每100步检查一次
        
    def record_training_metrics(self, gradient_norm: float, loss: float):
        """
        记录训练指标用于自适应调整
        
        Args:
            gradient_norm: 梯度范数
            loss: 损失值
        """
        self.gradient_norm_history.append(gradient_norm)
        self.loss_history.append(loss)
        
        # 保持历史记录在合理范围内
        if len(self.gradient_norm_history) > 1000:
            self.gradient_norm_history = self.gradient_norm_history[-500:]
            self.loss_history = self.loss_history[-500:]
    
    def should_adapt_accumulation(self) -> bool:
        """
        判断是否应该调整累积策略
        
        Returns:
            是否应该调整
        """
        if not self.adaptation_enabled:
            return False
        
        if len(self.gradient_norm_history) < self.adaptation_interval:
            return False
        
        # 检查梯度范数的稳定性
        recent_norms = self.gradient_norm_history[-self.adaptation_interval:]
        norm_std = np.std(recent_norms)
        norm_mean = np.mean(recent_norms)
        
        # 如果梯度范数变化很大，可能需要调整累积策略
        coefficient_of_variation = norm_std / norm_mean if norm_mean > 0 else 0
        
        return coefficient_of_variation > 0.5  # 变异系数大于0.5时考虑调整
    
    def adapt_accumulation_steps(self) -> Optional[int]:
        """
        自适应调整累积步数
        
        Returns:
            新的累积步数，如果不需要调整则返回None
        """
        if not self.should_adapt_accumulation():
            return None
        
        # 基于梯度范数历史调整策略
        recent_norms = self.gradient_norm_history[-self.adaptation_interval:]
        norm_mean = np.mean(recent_norms)
        
        # 如果梯度范数太小，增加累积步数
        # 如果梯度范数太大，减少累积步数
        if norm_mean < 0.1:
            new_steps = min(self.max_accumulation_steps, self.accumulation_steps + 1)
        elif norm_mean > 2.0:
            new_steps = max(self.min_accumulation_steps, self.accumulation_steps - 1)
        else:
            return None  # 不需要调整
        
        if new_steps != self.accumulation_steps:
            old_steps = self.accumulation_steps
            self.accumulation_steps = new_steps
            self.effective_batch_size = self.current_batch_size * new_steps
            
            print(f"🔄 自适应调整累积步数: {old_steps} → {new_steps}")
            print(f"   梯度范数均值: {norm_mean:.3f}")
            print(f"   新等效batch: {self.effective_batch_size}")
            
            return new_steps
        
        return None


if __name__ == "__main__":
    # 测试梯度累积管理器
    print("🧪 测试梯度累积管理器...")
    
    # 基础管理器测试
    manager = GradientAccumulationManager(target_effective_batch_size=16)
    
    # 测试不同的batch size配置
    test_batch_sizes = [4, 8, 12, 16, 24]
    
    print(f"\n📊 不同batch size的累积配置:")
    for batch_size in test_batch_sizes:
        accumulation_steps = manager.calculate_accumulation_steps(batch_size)
        config = manager.get_current_config()
        print(f"   batch_size={batch_size}: accumulation={accumulation_steps}, effective={config['effective_batch_size']}")
    
    # 测试梯度更新逻辑
    print(f"\n🔬 测试梯度更新逻辑 (batch_size=8, accumulation=2):")
    manager.calculate_accumulation_steps(8)
    
    for step in range(10):
        should_update = manager.should_update_gradients(step)
        loss_scale = manager.get_loss_scale()
        print(f"   步骤 {step}: 更新梯度={'✅' if should_update else '❌'}, 损失缩放={loss_scale:.3f}")
    
    # 显示统计信息
    print(f"\n📈 统计信息:")
    manager.print_status(detailed=True)
    
    # 测试自适应管理器
    print(f"\n🧪 测试自适应梯度累积管理器...")
    adaptive_manager = AdaptiveGradientAccumulation(target_effective_batch_size=16)
    adaptive_manager.calculate_accumulation_steps(8)
    
    # 模拟一些训练指标
    for i in range(150):
        # 模拟梯度范数和损失
        gradient_norm = np.random.normal(1.0, 0.5)
        loss = 3.0 - i * 0.01 + np.random.normal(0, 0.1)
        
        adaptive_manager.record_training_metrics(gradient_norm, loss)
        
        # 每50步检查一次自适应
        if i % 50 == 49:
            new_steps = adaptive_manager.adapt_accumulation_steps()
            if new_steps:
                print(f"   步骤 {i}: 累积步数调整为 {new_steps}")
    
    print("✅ 梯度累积管理器测试完成！")

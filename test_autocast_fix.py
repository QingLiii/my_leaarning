#!/usr/bin/env python3
"""
测试新的torch.amp.autocast用法是否正常工作
验证FutureWarning修复是否成功
"""

import torch
import warnings
import sys

def test_new_autocast():
    """测试新的autocast用法"""
    print("🧪 测试新的torch.amp.autocast用法...")
    
    # 捕获警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，跳过测试")
            return False
        
        # 创建测试数据
        device = 'cuda'
        x = torch.randn(2, 3, 224, 224, device=device)
        
        # 测试新的autocast用法
        try:
            with torch.amp.autocast('cuda'):
                y = x * 2.0
                z = torch.sum(y)
            
            print("✅ 新的torch.amp.autocast('cuda')用法正常工作")
            
            # 检查是否有FutureWarning
            future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
            autocast_warnings = [warning for warning in future_warnings if 'autocast' in str(warning.message)]
            
            if autocast_warnings:
                print("❌ 仍然存在autocast相关的FutureWarning:")
                for warning in autocast_warnings:
                    print(f"   {warning.message}")
                return False
            else:
                print("✅ 没有检测到autocast相关的FutureWarning")
                return True
                
        except Exception as e:
            print(f"❌ 新的autocast用法出错: {e}")
            return False

def test_new_gradscaler():
    """测试新的GradScaler用法"""
    print("\n🧪 测试新的torch.amp.GradScaler用法...")
    
    # 捕获警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            # 测试新的GradScaler用法
            scaler = torch.amp.GradScaler('cuda')
            print("✅ 新的torch.amp.GradScaler('cuda')用法正常工作")
            
            # 检查是否有FutureWarning
            future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
            gradscaler_warnings = [warning for warning in future_warnings if 'GradScaler' in str(warning.message)]
            
            if gradscaler_warnings:
                print("❌ 仍然存在GradScaler相关的FutureWarning:")
                for warning in gradscaler_warnings:
                    print(f"   {warning.message}")
                return False
            else:
                print("✅ 没有检测到GradScaler相关的FutureWarning")
                return True
                
        except Exception as e:
            print(f"❌ 新的GradScaler用法出错: {e}")
            return False

def test_old_autocast_warning():
    """测试旧的autocast用法是否会产生警告"""
    print("\n🧪 测试旧的torch.cuda.amp.autocast用法是否产生警告...")
    
    # 捕获警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，跳过测试")
            return True
        
        try:
            device = 'cuda'
            x = torch.randn(2, 3, 224, 224, device=device)
            
            # 使用旧的autocast用法
            with torch.cuda.amp.autocast():
                y = x * 2.0
                z = torch.sum(y)
            
            # 检查是否有FutureWarning
            future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
            autocast_warnings = [warning for warning in future_warnings if 'autocast' in str(warning.message)]
            
            if autocast_warnings:
                print("✅ 旧的用法确实产生了FutureWarning:")
                for warning in autocast_warnings:
                    print(f"   {warning.message}")
                return True
            else:
                print("⚠️ 旧的用法没有产生预期的FutureWarning")
                return False
                
        except Exception as e:
            print(f"❌ 旧的autocast用法出错: {e}")
            return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🔧 FutureWarning修复验证测试")
    print("=" * 60)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "=" * 60)
    
    # 运行测试
    test1_passed = test_new_autocast()
    test2_passed = test_new_gradscaler()
    test3_passed = test_old_autocast_warning()
    
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    print("=" * 60)
    
    print(f"✅ 新autocast用法测试: {'通过' if test1_passed else '失败'}")
    print(f"✅ 新GradScaler用法测试: {'通过' if test2_passed else '失败'}")
    print(f"✅ 旧用法警告验证: {'通过' if test3_passed else '失败'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    
    if all_passed:
        print("\n🎉 所有测试通过！FutureWarning修复成功！")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查修复情况")
        return 1

if __name__ == "__main__":
    sys.exit(main())

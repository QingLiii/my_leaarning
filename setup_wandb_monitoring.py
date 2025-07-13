#!/usr/bin/env python3
"""
WandB监控系统安装和配置脚本
自动安装依赖、配置环境、验证功能
"""

import subprocess
import sys
import os
import importlib


def run_command(command, description=""):
    """运行命令并处理错误"""
    print(f"🔧 {description}")
    print(f"   执行: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"   输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 错误: {e}")
        if e.stderr:
            print(f"   错误信息: {e.stderr.strip()}")
        return False


def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安装")
        return False


def install_dependencies():
    """安装必要的依赖包"""
    print("📦 安装WandB监控依赖包...")
    
    # 必需的包列表
    packages = [
        ("wandb", "wandb"),
        ("GPUtil", "GPUtil"),
        ("nvidia-ml-py3", "nvidia_ml_py3"),
        ("psutil", "psutil"),
    ]
    
    # 检查已安装的包
    missing_packages = []
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    # 安装缺失的包
    if missing_packages:
        print(f"\n🔧 需要安装的包: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            success = run_command(
                f"pip install {package}",
                f"安装 {package}"
            )
            if not success:
                print(f"⚠️ {package} 安装失败，请手动安装")
                return False
    else:
        print("✅ 所有依赖包已安装")
    
    return True


def setup_wandb_config():
    """配置WandB"""
    print("\n🔑 配置WandB...")
    
    api_key = "68c9ce2a167992d06678c4fdc0d1075b5dfff922"
    
    # 设置API key
    success = run_command(
        f"wandb login {api_key}",
        "设置WandB API密钥"
    )
    
    if success:
        print("✅ WandB配置完成")
    else:
        print("⚠️ WandB配置失败，请手动运行: wandb login")
    
    return success


def verify_gpu_monitoring():
    """验证GPU监控功能"""
    print("\n🖥️ 验证GPU监控功能...")
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        
        if gpus:
            print(f"✅ 检测到 {len(gpus)} 个GPU:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                print(f"   显存: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
                print(f"   利用率: {gpu.load * 100:.1f}%")
                print(f"   温度: {gpu.temperature}°C")
        else:
            print("⚠️ 未检测到GPU")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU监控验证失败: {e}")
        return False


def verify_nvidia_ml():
    """验证NVIDIA ML监控功能"""
    print("\n🔍 验证NVIDIA ML监控功能...")
    
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        
        device_count = nvml.nvmlDeviceGetCount()
        print(f"✅ NVIDIA ML检测到 {device_count} 个设备")
        
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
            print(f"   设备 {i}: {name}")
            
            # 尝试获取功耗信息
            try:
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                print(f"   功耗: {power:.1f}W")
            except:
                print("   功耗: 不支持")
        
        return True
        
    except Exception as e:
        print(f"⚠️ NVIDIA ML监控不可用: {e}")
        print("   这不会影响基础GPU监控功能")
        return False


def create_test_script():
    """创建快速测试脚本"""
    print("\n📝 创建快速测试脚本...")
    
    test_script = """#!/usr/bin/env python3
# 快速WandB监控测试
import sys
sys.path.append('R2Gen-main')

from modules.wandb_logger import WandBLogger
import time

def quick_test():
    print("🧪 快速WandB测试...")
    
    logger = WandBLogger(project_name="R2Gen-Quick-Test")
    
    config = {
        'test_type': 'quick_verification',
        'batch_size': 4,
        'learning_rate': 0.001
    }
    
    logger.init_run(config, run_name="quick_test")
    
    # 记录一些测试数据
    for i in range(5):
        logger.log_training_metrics(
            epoch=1,
            batch_idx=i,
            loss=3.0 - i * 0.1,
            learning_rate=0.001
        )
        time.sleep(0.5)
    
    logger.finish()
    print("✅ 快速测试完成！")

if __name__ == '__main__':
    quick_test()
"""
    
    with open('quick_wandb_test.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✅ 快速测试脚本已创建: quick_wandb_test.py")


def main():
    """主安装流程"""
    print("🚀 WandB监控系统安装和配置")
    print("=" * 50)
    
    success_steps = 0
    total_steps = 5
    
    # 步骤1: 安装依赖
    if install_dependencies():
        success_steps += 1
        print("✅ 步骤1完成: 依赖安装")
    else:
        print("❌ 步骤1失败: 依赖安装")
    
    # 步骤2: 配置WandB
    if setup_wandb_config():
        success_steps += 1
        print("✅ 步骤2完成: WandB配置")
    else:
        print("❌ 步骤2失败: WandB配置")
    
    # 步骤3: 验证GPU监控
    if verify_gpu_monitoring():
        success_steps += 1
        print("✅ 步骤3完成: GPU监控验证")
    else:
        print("❌ 步骤3失败: GPU监控验证")
    
    # 步骤4: 验证NVIDIA ML（可选）
    if verify_nvidia_ml():
        success_steps += 1
        print("✅ 步骤4完成: NVIDIA ML验证")
    else:
        print("⚠️ 步骤4跳过: NVIDIA ML验证（可选功能）")
        success_steps += 1  # 这是可选功能，不影响总体成功
    
    # 步骤5: 创建测试脚本
    try:
        create_test_script()
        success_steps += 1
        print("✅ 步骤5完成: 测试脚本创建")
    except Exception as e:
        print(f"❌ 步骤5失败: 测试脚本创建 - {e}")
    
    # 总结
    print("\n" + "=" * 50)
    print(f"📊 安装完成: {success_steps}/{total_steps} 步骤成功")
    
    if success_steps >= 4:  # 至少4个步骤成功（NVIDIA ML是可选的）
        print("🎉 WandB监控系统安装成功！")
        print("\n📋 下一步操作:")
        print("1. 运行快速测试: python quick_wandb_test.py")
        print("2. 运行完整测试: python test_wandb_integration.py --test-all")
        print("3. 开始使用增强版训练器")
        
        print("\n🔧 使用方法:")
        print("```python")
        print("from modules.wandb_logger import WandBLogger")
        print("from modules.enhanced_trainer import EnhancedTrainer")
        print("")
        print("# 创建WandB logger")
        print("wandb_logger = WandBLogger()")
        print("")
        print("# 在trainer中使用")
        print("trainer = EnhancedTrainer(..., wandb_logger=wandb_logger)")
        print("```")
        
        return True
    else:
        print("⚠️ 安装过程中遇到问题，请检查错误信息")
        print("\n🔧 手动安装命令:")
        print("pip install wandb GPUtil nvidia-ml-py3 psutil")
        print("wandb login 68c9ce2a167992d06678c4fdc0d1075b5dfff922")
        
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

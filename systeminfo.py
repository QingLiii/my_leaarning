#!/usr/bin/env python3
"""
系统信息查看工具
检查 CUDA、PyTorch 版本及可用性
"""

import sys
import platform
import subprocess

def get_python_info():
    """获取 Python 版本信息"""
    print("=" * 50)
    print("Python 信息")
    print("=" * 50)
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    print(f"平台: {platform.platform()}")
    print()

def get_cuda_info():
    """获取 CUDA 版本信息"""
    print("=" * 50)
    print("CUDA 信息")
    print("=" * 50)
    
    try:
        # 检查 nvidia-smi 命令
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version: ')[1].split()[0]
                    print(f"NVIDIA 驱动 CUDA 版本: {cuda_version}")
                    break
        else:
            print("未找到 NVIDIA 驱动或 nvidia-smi 命令")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("无法执行 nvidia-smi 命令 (可能没有安装 NVIDIA 驱动)")
    
    try:
        # 检查 nvcc 版本
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line:
                    print(f"NVCC 版本: {line.strip()}")
                    break
        else:
            print("未找到 NVCC (CUDA 开发工具包)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("未安装 CUDA 开发工具包 (nvcc)")
    
    print()

def get_torch_info():
    """获取 PyTorch 信息"""
    print("=" * 50)
    print("PyTorch 信息")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"PyTorch 路径: {torch.__file__}")
        
        # CUDA 可用性
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 是否可用: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA 版本 (PyTorch): {torch.version.cuda}")
            print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            
            # 列出所有 GPU
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # 当前 GPU
            if torch.cuda.device_count() > 0:
                current_device = torch.cuda.current_device()
                print(f"当前 GPU: {current_device}")
        else:
            print("PyTorch 未检测到可用的 CUDA GPU")
        
        # MPS (Apple Silicon) 支持
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"MPS (Apple Silicon) 可用: True")
        
    except ImportError:
        print("PyTorch 未安装")
        print("安装命令:")
        print("  CPU 版本: pip install torch")
        print("  CUDA 版本: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    print()

def get_other_ml_libraries():
    """检查其他机器学习库"""
    print("=" * 50)
    print("其他机器学习库")
    print("=" * 50)
    
    libraries = [
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tensorflow',
        'transformers',
        'opencv-cv2'
    ]
    
    for lib in libraries:
        try:
            if lib == 'opencv-cv2':
                import cv2
                print(f"OpenCV: {cv2.__version__}")
            elif lib == 'scikit-learn':
                import sklearn
                print(f"scikit-learn: {sklearn.__version__}")
            else:
                module = __import__(lib)
                if hasattr(module, '__version__'):
                    print(f"{lib}: {module.__version__}")
                else:
                    print(f"{lib}: 已安装 (版本未知)")
        except ImportError:
            print(f"{lib}: 未安装")
    
    print()

def main():
    """主函数"""
    print("系统信息检查工具")
    print("检查时间:", end=" ")
    subprocess.run(['date'])
    print()
    
    get_python_info()
    get_cuda_info()
    get_torch_info()
    get_other_ml_libraries()
    
    print("=" * 50)
    print("检查完成")
    print("=" * 50)

if __name__ == "__main__":
    main()

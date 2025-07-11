#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IU X-ray 数据集下载和处理脚本

该脚本提供了下载和处理IU X-ray数据集的功能。
作者: 生物医学工程专业学习项目
用途: 医学影像报告生成研究
"""

import os
import zipfile
import requests
from pathlib import Path
import json

def check_kaggle_config():
    """
    检查Kaggle API配置是否存在
    
    Returns:
        bool: 如果配置存在返回True，否则返回False
    """
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print("✓ 找到Kaggle API配置文件")
        return True
    else:
        print("✗ 未找到Kaggle API配置文件")
        print("请按照README.md中的说明配置Kaggle API")
        return False

def download_with_kaggle_api():
    """
    使用Kaggle API下载数据集
    
    Returns:
        bool: 下载成功返回True，否则返回False
    """
    try:
        import kaggle
        
        # 设置下载路径
        download_path = Path(__file__).parent / 'iu_xray'
        download_path.mkdir(exist_ok=True)
        
        print("开始使用Kaggle API下载数据集...")
        kaggle.api.dataset_download_files(
            'jiangzhiyan/iu-xray',
            path=str(download_path),
            unzip=True
        )
        
        print("✓ 数据集下载完成！")
        return True
        
    except Exception as e:
        print(f"✗ Kaggle API下载失败: {e}")
        return False

def manual_download_instructions():
    """
    显示手动下载说明
    """
    print("\n=== 手动下载说明 ===")
    print("1. 访问: https://www.kaggle.com/datasets/jiangzhiyan/iu-xray")
    print("2. 点击 'Download' 按钮下载数据集")
    print("3. 将下载的zip文件解压到以下目录:")
    print(f"   {Path(__file__).parent / 'iu_xray'}")
    print("4. 解压完成后运行此脚本验证数据集")

def verify_dataset():
    """
    验证数据集是否正确下载和解压
    
    Returns:
        bool: 验证成功返回True，否则返回False
    """
    dataset_path = Path(__file__).parent / 'iu_xray'
    
    if not dataset_path.exists():
        print("✗ 数据集目录不存在")
        return False
    
    # 检查是否有文件
    files = list(dataset_path.glob('*'))
    if len(files) == 0:
        print("✗ 数据集目录为空")
        return False
    
    print(f"✓ 找到 {len(files)} 个文件/目录")
    
    # 显示目录内容
    print("\n数据集内容:")
    for file in files[:10]:  # 只显示前10个
        if file.is_file():
            size = file.stat().st_size / (1024*1024)  # MB
            print(f"  📄 {file.name} ({size:.1f} MB)")
        else:
            print(f"  📁 {file.name}/")
    
    if len(files) > 10:
        print(f"  ... 还有 {len(files) - 10} 个文件")
    
    return True

def create_dataset_info():
    """
    创建数据集信息文件
    """
    dataset_path = Path(__file__).parent / 'iu_xray'
    info_file = dataset_path / 'dataset_info.json'
    
    info = {
        "name": "IU X-ray Dataset",
        "source": "https://www.kaggle.com/datasets/jiangzhiyan/iu-xray",
        "description": "印第安纳大学胸部X光数据集，包含胸部X光图像和对应的放射学报告",
        "purpose": "医学影像报告生成研究",
        "downloaded_by": "生物医学工程专业学习项目"
    }
    
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 创建数据集信息文件: {info_file}")

def main():
    """
    主函数：协调整个下载和验证过程
    """
    print("=== IU X-ray 数据集下载工具 ===")
    print("用于生物医学工程专业 - 医学影像报告生成研究\n")
    
    # 首先检查数据集是否已存在
    if verify_dataset():
        print("\n数据集已存在且验证通过！")
        create_dataset_info()
        return
    
    # 尝试使用Kaggle API下载
    if check_kaggle_config():
        if download_with_kaggle_api():
            verify_dataset()
            create_dataset_info()
            return
    
    # 如果API下载失败，显示手动下载说明
    manual_download_instructions()
    print("\n下载完成后，请再次运行此脚本验证数据集。")

if __name__ == "__main__":
    main()
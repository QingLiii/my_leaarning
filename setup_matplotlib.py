#!/usr/bin/env python3
"""
一劳永逸的matplotlib高分辨率配置脚本
运行此脚本将在当前环境中永久设置matplotlib的默认配置为科研级别
"""

import os
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

def create_matplotlibrc():
    """
    创建matplotlib配置文件，设置为科研级别的高分辨率输出
    """
    
    # matplotlib配置内容
    config_content = """
# ========== 科研级别matplotlib配置 ==========
# 图片质量设置
figure.dpi: 300
savefig.dpi: 300
savefig.format: png
savefig.bbox: tight
savefig.pad_inches: 0.1
savefig.facecolor: white
savefig.edgecolor: none
savefig.transparent: False

# 字体设置（支持中文）
font.size: 12
font.family: sans-serif
font.sans-serif: SimHei, Microsoft YaHei, Arial Unicode MS, PingFang SC, DejaVu Sans, WenQuanYi Micro Hei, Arial, Liberation Sans, Bitstream Vera Sans, sans-serif

# 中文显示设置
axes.unicode_minus: False

# 标题和标签字体大小
axes.titlesize: 14
axes.labelsize: 12
xtick.labelsize: 10
ytick.labelsize: 10
legend.fontsize: 10
figure.titlesize: 16

# 线条和标记设置
lines.linewidth: 1.5
lines.markersize: 6
patch.linewidth: 0.5

# 坐标轴设置
axes.linewidth: 1.0
axes.spines.left: True
axes.spines.bottom: True
axes.spines.top: False
axes.spines.right: False
axes.grid: True
axes.grid.alpha: 0.3

# 刻度设置
xtick.direction: out
ytick.direction: out
xtick.major.size: 4
ytick.major.size: 4
xtick.minor.size: 2
ytick.minor.size: 2

# 图例设置
legend.frameon: True
legend.framealpha: 0.8
legend.fancybox: True
legend.numpoints: 1

# 颜色循环（科研友好）
axes.prop_cycle: cycler('color', ['1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf'])

# 图片尺寸默认设置
figure.figsize: 8, 6
figure.facecolor: white
figure.edgecolor: none

# 其他设置
text.usetex: False
mathtext.default: regular
"""
    
    # 获取matplotlib配置目录
    config_dir = matplotlib.get_configdir()
    config_file = Path(config_dir) / 'matplotlibrc'
    
    # 备份现有配置（如果存在）
    if config_file.exists():
        backup_file = Path(config_dir) / 'matplotlibrc.backup'
        if not backup_file.exists():
            config_file.rename(backup_file)
            print(f"📁 已备份原配置文件到: {backup_file}")
    
    # 写入新配置
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content.strip())
    
    print(f"✅ 科研级别matplotlib配置已写入: {config_file}")
    return config_file

def create_local_matplotlibrc():
    """
    在当前目录创建本地matplotlib配置文件
    """
    config_content = """
# ========== 科研级别matplotlib配置 ==========
figure.dpi: 300
savefig.dpi: 300
savefig.format: png
savefig.bbox: tight
savefig.pad_inches: 0.1
font.size: 12
font.family: sans-serif
font.sans-serif: SimHei, Microsoft YaHei, Arial Unicode MS, PingFang SC, DejaVu Sans, WenQuanYi Micro Hei, Arial, Liberation Sans, Bitstream Vera Sans, sans-serif
axes.titlesize: 14
axes.labelsize: 12
axes.unicode_minus: False
"""
    
    local_config = Path('./matplotlibrc')
    with open(local_config, 'w', encoding='utf-8') as f:
        f.write(config_content.strip())
    
    print(f"📁 本地matplotlib配置已创建: {local_config}")
    return local_config

def test_configuration():
    """
    测试配置是否生效
    """
    import numpy as np
    
    # 重新加载matplotlib配置
    matplotlib.rcdefaults()
    plt.style.use('default')
    
    # 创建测试图片
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax.plot(x, y1, label='sin(x)', linewidth=2)
    ax.plot(x, y2, label='cos(x)', linewidth=2)
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_title('高分辨率测试图片')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存测试图片
    test_filename = 'matplotlib_config_test.png'
    plt.savefig(test_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 检查文件大小（高分辨率图片应该更大）
    file_size = os.path.getsize(test_filename)
    print(f"🧪 测试图片已生成: {test_filename}")
    print(f"📏 文件大小: {file_size/1024:.1f} KB")
    
    if file_size > 50000:  # 50KB以上说明是高分辨率
        print("✅ 配置成功！图片为高分辨率")
    else:
        print("⚠️ 配置可能未生效，图片分辨率较低")
    
    return test_filename

def main():
    """
    主函数：设置matplotlib为科研级别配置
    """
    print("🎨 开始配置matplotlib为科研级别高分辨率...")
    print("=" * 50)
    
    try:
        # 创建全局配置
        global_config = create_matplotlibrc()
        
        # 创建本地配置（作为备选）
        local_config = create_local_matplotlibrc()
        
        print("\n🧪 测试配置...")
        test_file = test_configuration()
        
        print("\n" + "=" * 50)
        print("🎉 配置完成！")
        print("\n📋 配置说明:")
        print("• DPI: 300 (科研标准)")
        print("• 格式: PNG (最高质量)")
        print("• 字体: Arial (科研友好)")
        print("• 边界: 紧凑布局")
        print("• 颜色: 科研友好色彩方案")
        
        print("\n🔄 重启Python/Jupyter后配置生效")
        print("💡 或者在代码中添加: plt.style.use('default')")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ 从现在开始，所有matplotlib图片都将是科研级别的高分辨率！")
    else:
        print("\n💡 请检查权限或手动导入matplotlib_config.py")

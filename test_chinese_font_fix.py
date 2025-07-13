#!/usr/bin/env python3
"""
测试中文字体修复效果的脚本
验证matplotlib是否能正确显示中文字符
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

def test_chinese_font_display():
    """
    测试中文字体显示效果
    """
    print("🧪 测试中文字体显示...")
    
    # 导入并应用配置
    try:
        from matplotlib_config import setup_high_quality_plots
        setup_high_quality_plots()
        print("✅ 已加载matplotlib_config配置")
    except ImportError:
        print("⚠️ matplotlib_config.py未找到，使用手动配置")
        # 手动设置中文字体
        mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 
                                           'PingFang SC', 'DejaVu Sans', 'WenQuanYi Micro Hei',
                                           'Arial', 'Liberation Sans', 'sans-serif']
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['savefig.dpi'] = 300
    
    # 检查当前字体设置
    print(f"📝 当前字体设置: {mpl.rcParams['font.sans-serif']}")
    print(f"🔤 Unicode负号设置: {mpl.rcParams['axes.unicode_minus']}")
    
    # 创建测试图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 测试1：基本中文标签
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax1.plot(x, y1, label='正弦函数 sin(x)', linewidth=2)
    ax1.plot(x, y2, label='余弦函数 cos(x)', linewidth=2)
    ax1.set_xlabel('横坐标 (X轴)')
    ax1.set_ylabel('纵坐标 (Y轴)')
    ax1.set_title('测试1: 基本中文标签显示')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 测试2：包含数字和符号的中文
    categories = ['类别A', '类别B', '类别C', '类别D', '类别E']
    values = [23, 45, 56, 78, 32]
    
    ax2.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax2.set_xlabel('产品类别')
    ax2.set_ylabel('销售数量 (万件)')
    ax2.set_title('测试2: 柱状图中文标签')
    
    # 在柱子上添加数值标签
    for i, v in enumerate(values):
        ax2.text(i, v + 1, f'{v}万', ha='center', va='bottom')
    
    # 测试3：负数和特殊字符
    x3 = np.linspace(-5, 5, 100)
    y3 = x3**2 - 10
    
    ax3.plot(x3, y3, 'r-', linewidth=2, label='抛物线: y = x² - 10')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('X轴 (包含负数)')
    ax3.set_ylabel('Y轴 (包含负数)')
    ax3.set_title('测试3: 负数显示测试')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 测试4：复杂中文文本
    data = np.random.randn(1000)
    ax4.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('数值范围')
    ax4.set_ylabel('频次统计')
    ax4.set_title('测试4: 随机数据分布直方图\n（包含换行的中文标题）')
    
    # 添加统计信息
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax4.text(0.02, 0.98, f'均值: {mean_val:.2f}\n标准差: {std_val:.2f}', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存测试图片
    test_filename = 'chinese_font_test_fixed.png'
    plt.savefig(test_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 检查文件
    if os.path.exists(test_filename):
        file_size = os.path.getsize(test_filename)
        print(f"✅ 测试图片已生成: {test_filename}")
        print(f"📏 文件大小: {file_size/1024:.1f} KB")
        
        # 简单的视觉检查提示
        print("\n🔍 请检查生成的图片:")
        print("• 所有中文字符应该正常显示，不应该有方框 □")
        print("• 负号应该正确显示为 - 而不是其他符号")
        print("• 图片应该是高分辨率 (300 DPI)")
        
        return True
    else:
        print("❌ 测试图片生成失败")
        return False

def check_available_fonts():
    """
    检查系统中可用的中文字体
    """
    print("\n🔤 检查系统可用字体...")
    
    from matplotlib.font_manager import FontManager
    fm = FontManager()
    
    # 查找中文字体
    chinese_fonts = []
    target_fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 
                   'PingFang SC', 'DejaVu Sans', 'WenQuanYi Micro Hei']
    
    for font in fm.ttflist:
        font_name = font.name
        if any(target in font_name for target in target_fonts):
            if font_name not in chinese_fonts:
                chinese_fonts.append(font_name)
    
    if chinese_fonts:
        print("✅ 找到以下中文字体:")
        for font in chinese_fonts:
            print(f"  • {font}")
    else:
        print("⚠️ 未找到常见的中文字体")
        print("💡 建议安装以下字体之一:")
        for font in target_fonts:
            print(f"  • {font}")
    
    return chinese_fonts

def main():
    """
    主测试函数
    """
    print("🎨 开始测试中文字体修复效果...")
    print("=" * 50)
    
    # 检查可用字体
    available_fonts = check_available_fonts()
    
    # 运行显示测试
    success = test_chinese_font_display()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 测试完成！")
        print("\n📋 修复说明:")
        print("• 已在matplotlib配置中添加中文字体支持")
        print("• 设置了正确的Unicode负号显示")
        print("• 支持多种操作系统的常见中文字体")
        
        if not available_fonts:
            print("\n⚠️ 注意: 如果仍有显示问题，请安装中文字体")
    else:
        print("❌ 测试失败，请检查配置")
    
    return success

if __name__ == "__main__":
    main()

"""
matplotlib高分辨率科研级别配置
用于确保所有图片都是科研文献可用的高质量图片
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def setup_high_quality_plots():
    """
    设置matplotlib为科研级别的高质量图片输出
    
    配置包括：
    - 高DPI设置（300 DPI，科研标准）
    - 矢量图格式支持
    - 高质量字体设置
    - 优化的颜色和样式
    """
    
    # ========== 基础质量设置 ==========
    # 设置默认DPI为300（科研标准）
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    
    # 设置默认图片格式为PNG，质量最高
    mpl.rcParams['savefig.format'] = 'png'
    mpl.rcParams['savefig.bbox'] = 'tight'  # 紧凑边界
    mpl.rcParams['savefig.pad_inches'] = 0.1  # 适当边距
    
    # ========== 字体设置 ==========
    # 设置字体大小和类型
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['figure.titlesize'] = 16
    
    # 设置字体族（优先使用无衬线字体）
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 
                                       'Bitstream Vera Sans', 'sans-serif']
    
    # ========== 线条和标记设置 ==========
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 6
    mpl.rcParams['patch.linewidth'] = 0.5
    
    # ========== 坐标轴设置 ==========
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.bottom'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.grid.alpha'] = 0.3
    
    # ========== 刻度设置 ==========
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['ytick.major.size'] = 4
    mpl.rcParams['xtick.minor.size'] = 2
    mpl.rcParams['ytick.minor.size'] = 2
    
    # ========== 图例设置 ==========
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.8
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['legend.numpoints'] = 1
    
    # ========== 颜色设置 ==========
    # 使用科研友好的颜色方案
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 橄榄色
        '#17becf'   # 青色
    ])
    
    print("✅ matplotlib已配置为科研级别高质量输出")
    print("📊 DPI: 300 | 格式: PNG | 字体: Arial")

def save_high_quality_figure(filename, **kwargs):
    """
    保存高质量图片的便捷函数
    
    Args:
        filename (str): 文件名
        **kwargs: 传递给plt.savefig的额外参数
    """
    default_kwargs = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'facecolor': 'white',
        'edgecolor': 'none',
        'transparent': False
    }
    
    # 合并用户参数和默认参数
    default_kwargs.update(kwargs)
    
    plt.savefig(filename, **default_kwargs)
    print(f"💾 高质量图片已保存: {filename}")

def create_publication_figure(figsize=(10, 6), **kwargs):
    """
    创建科研发表级别的图片
    
    Args:
        figsize (tuple): 图片尺寸 (宽, 高) 英寸
        **kwargs: 传递给plt.figure的额外参数
    
    Returns:
        matplotlib.figure.Figure: 图片对象
    """
    default_kwargs = {
        'figsize': figsize,
        'dpi': 300,
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    
    default_kwargs.update(kwargs)
    
    fig = plt.figure(**default_kwargs)
    return fig

# 科研常用的图片尺寸（英寸）
FIGURE_SIZES = {
    'single_column': (3.5, 2.625),      # 单栏图片
    'double_column': (7, 5.25),         # 双栏图片
    'full_page': (7, 9),                # 全页图片
    'square': (6, 6),                   # 正方形图片
    'wide': (10, 4),                    # 宽图片
    'tall': (6, 8),                     # 高图片
    'presentation': (12, 8),            # 演示用图片
}

def get_figure_size(size_name):
    """
    获取预定义的图片尺寸
    
    Args:
        size_name (str): 尺寸名称
        
    Returns:
        tuple: (宽, 高) 英寸
    """
    return FIGURE_SIZES.get(size_name, FIGURE_SIZES['double_column'])

# 自动应用配置
if __name__ == "__main__":
    setup_high_quality_plots()
else:
    # 当模块被导入时自动应用配置
    setup_high_quality_plots()

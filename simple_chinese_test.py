#!/usr/bin/env python3
"""
简化的中文字体测试脚本
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 强制使用WenQuanYi Micro Hei字体（系统中可用的中文字体）
mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300

print("🧪 创建简化的中文字体测试...")
print(f"📝 使用字体: {mpl.rcParams['font.sans-serif'][0]}")

# 创建简单的测试图表
fig, ax = plt.subplots(figsize=(8, 6))

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

ax.plot(x, y1, label='正弦函数', linewidth=2)
ax.plot(x, y2, label='余弦函数', linewidth=2)
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_title('中文字体测试图片')
ax.legend()
ax.grid(True, alpha=0.3)

# 保存图片
plt.savefig('simple_chinese_test.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 测试图片已生成: simple_chinese_test.png")
print("🔍 请检查图片中的中文字符是否正常显示")

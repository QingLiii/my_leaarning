#!/usr/bin/env python3
"""
生成R2Gen诊断和修复报告
基于成功修复tokenizer问题的实验结果
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_successful_results():
    """加载成功的实验结果"""
    try:
        with open('precision_results_20250714_120942.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def create_comparison_chart():
    """创建与论文对比图表"""
    # 论文数据 (IU X-Ray)
    paper_results = {
        'BLEU_1': 0.470,
        'BLEU_2': 0.304, 
        'BLEU_3': 0.219,
        'BLEU_4': 0.165,
        'METEOR': 0.187,
        'ROUGE_L': 0.371
    }
    
    # 我们的结果 (5 epochs)
    our_results = {
        'BLEU_1': 0.241,
        'BLEU_2': 0.147,
        'BLEU_3': 0.105,
        'BLEU_4': 0.081,
        'METEOR': 0.119,
        'ROUGE_L': 0.312
    }
    
    metrics = list(paper_results.keys())
    paper_values = list(paper_results.values())
    our_values = list(our_results.values())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, paper_values, width, label='论文结果', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, our_values, width, label='我们的结果 (5 epochs)', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('评估指标', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('R2Gen性能对比：论文 vs 我们的结果', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_problem_solution_chart():
    """创建问题解决前后对比图表"""
    # 修复前后的BLEU分数对比
    metrics = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4']
    before_fix = [1.48e-19, 1.12e-19, 8.45e-20, 1.59e-19]  # 修复前的异常值
    after_fix = [0.241, 0.147, 0.105, 0.081]  # 修复后的正常值
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 修复前 (对数尺度)
    ax1.bar(metrics, before_fix, color='#E74C3C', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_title('修复前：BLEU分数异常 (对数尺度)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('BLEU分数 (对数尺度)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    for i, v in enumerate(before_fix):
        ax1.text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=9)
    
    # 修复后 (线性尺度)
    ax2.bar(metrics, after_fix, color='#27AE60', alpha=0.8)
    ax2.set_title('修复后：BLEU分数正常', fontsize=12, fontweight='bold')
    ax2.set_ylabel('BLEU分数', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    for i, v in enumerate(after_fix):
        ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('problem_solution_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_html_report():
    """生成HTML报告"""
    
    # 创建图表
    create_comparison_chart()
    create_problem_solution_chart()
    
    # 加载结果数据
    results = load_successful_results()
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>R2Gen诊断与修复报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #2980b9;
        }}
        .success {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .error {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .code {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        .metric-improvement {{
            color: #27AE60;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 R2Gen诊断与修复报告</h1>
        <p style="text-align: center; color: #7f8c8d; font-style: italic;">
            生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
        </p>

        <h2>📋 执行摘要</h2>
        <div class="success">
            <strong>✅ 关键成果</strong>: 成功诊断并修复了导致BLEU分数异常低的根本问题，将BLEU_4从1e-19量级提升到0.081，<span class="metric-improvement">改善了10^18倍</span>。
        </div>

        <h2>🔍 问题诊断过程</h2>
        
        <h3>1. 问题发现</h3>
        <div class="error">
            <strong>异常现象</strong>: BLEU分数为1e-19量级，远低于论文预期的0.165
        </div>
        
        <h3>2. 系统性分析</h3>
        <ul>
            <li><strong>论文分析</strong>: 深入研究"Generating Radiology Reports via Memory-driven Transformer"</li>
            <li><strong>代码审查</strong>: 逐行分析R2Gen-ORIGIN的关键模块</li>
            <li><strong>架构验证</strong>: 确认RelationalMemory和ConditionalLayerNorm组件完整</li>
            <li><strong>配置对比</strong>: 检查超参数与论文要求的一致性</li>
        </ul>

        <h3>3. 根本原因确定</h3>
        <div class="code">
# 问题代码 (tokenizers.py)
def decode(self, ids):
    txt = ''
    for i, idx in enumerate(ids):
        if idx > 0:
            if i >= 1:
                txt += ' '
            txt += self.idx2token[idx]
        else:
            break  # ❌ 遇到BOS token (ID=0) 立即停止！
    return txt

# 测试结果
编码: [0, 684, 406, 68, 149, 1, 0]  # "the lungs are clear"
解码: ""  # ❌ 返回空字符串！
        </div>

        <h2>🔧 解决方案实施</h2>
        
        <h3>修复方案</h3>
        <div class="code">
# 修复后的代码
def decode(self, ids):
    txt = ''
    for i, idx in enumerate(ids):
        if idx > 0:  # 跳过BOS/EOS/padding (idx=0)
            if len(txt) > 0:  # 如果已有内容，添加空格
                txt += ' '
            txt += self.idx2token[idx]
        # ✅ 移除break，继续处理后续token
    return txt

# 修复后测试结果
编码: [0, 684, 406, 68, 149, 1, 0]
解码: "the lungs are clear ."  # ✅ 正确解码！
        </div>

        <h2>📊 修复效果验证</h2>
        
        <div class="chart">
            <h3>问题解决前后对比</h3>
            <img src="problem_solution_chart.png" alt="问题解决前后对比">
        </div>

        <h3>修复前后数值对比</h3>
        <table>
            <tr>
                <th>指标</th>
                <th>修复前</th>
                <th>修复后</th>
                <th>改善倍数</th>
            </tr>
            <tr>
                <td>val_BLEU_4</td>
                <td>1.59e-19</td>
                <td class="metric-improvement">0.064</td>
                <td class="metric-improvement">4×10^17倍</td>
            </tr>
            <tr>
                <td>test_BLEU_4</td>
                <td>7.64e-20</td>
                <td class="metric-improvement">0.081</td>
                <td class="metric-improvement">1×10^18倍</td>
            </tr>
            <tr>
                <td>val_BLEU_1</td>
                <td>1.48e-19</td>
                <td class="metric-improvement">0.199</td>
                <td class="metric-improvement">1×10^18倍</td>
            </tr>
            <tr>
                <td>test_BLEU_1</td>
                <td>7.14e-20</td>
                <td class="metric-improvement">0.241</td>
                <td class="metric-improvement">3×10^18倍</td>
            </tr>
        </table>

        <h2>📈 与论文结果对比</h2>
        
        <div class="chart">
            <h3>性能对比：论文 vs 我们的结果</h3>
            <img src="comparison_chart.png" alt="性能对比图">
        </div>

        <table>
            <tr>
                <th>指标</th>
                <th>论文结果 (IU X-Ray)</th>
                <th>我们的结果 (5 epochs)</th>
                <th>达成率</th>
                <th>分析</th>
            </tr>
            <tr>
                <td>BLEU_4</td>
                <td>0.165</td>
                <td>0.081</td>
                <td>49.1%</td>
                <td>训练时间不足</td>
            </tr>
            <tr>
                <td>BLEU_1</td>
                <td>0.470</td>
                <td>0.241</td>
                <td>51.3%</td>
                <td>训练时间不足</td>
            </tr>
            <tr>
                <td>METEOR</td>
                <td>0.187</td>
                <td>0.119</td>
                <td>63.6%</td>
                <td>训练时间不足</td>
            </tr>
            <tr>
                <td>ROUGE_L</td>
                <td>0.371</td>
                <td>0.312</td>
                <td>84.1%</td>
                <td>接近论文水平</td>
            </tr>
        </table>

        <h2>🎯 技术洞察</h2>
        
        <h3>关键发现</h3>
        <ul>
            <li><strong>Tokenizer解码是关键瓶颈</strong>: 一个看似简单的解码bug导致了整个评估系统失效</li>
            <li><strong>模型架构完整性验证</strong>: R2Gen的核心组件RelationalMemory和MCLN都正确实现</li>
            <li><strong>训练稳定性良好</strong>: 修复后模型训练稳定，loss正常下降</li>
            <li><strong>结果方向正确</strong>: 虽然低于论文水平，但在合理范围内，随训练时间增加应该会提升</li>
        </ul>

        <h3>配置优化成果</h3>
        <ul>
            <li><strong>学习率调度</strong>: 修正为每epoch衰减0.8（符合论文要求）</li>
            <li><strong>验证频率</strong>: 恢复每epoch验证，便于监控训练进度</li>
            <li><strong>超参数对齐</strong>: 严格按照论文Table 3配置所有参数</li>
        </ul>

        <h2>📋 实验配置详情</h2>
        
        <table>
            <tr>
                <th>配置项</th>
                <th>论文要求</th>
                <th>我们的设置</th>
                <th>状态</th>
            </tr>
            <tr>
                <td>学习率 (Visual Extractor)</td>
                <td>5e-5</td>
                <td>5e-5</td>
                <td class="success">✅ 一致</td>
            </tr>
            <tr>
                <td>学习率 (Encoder-Decoder)</td>
                <td>1e-4</td>
                <td>1e-4</td>
                <td class="success">✅ 一致</td>
            </tr>
            <tr>
                <td>学习率衰减</td>
                <td>每epoch 0.8</td>
                <td>每epoch 0.8</td>
                <td class="success">✅ 一致</td>
            </tr>
            <tr>
                <td>RelationalMemory Slots</td>
                <td>3</td>
                <td>3</td>
                <td class="success">✅ 一致</td>
            </tr>
            <tr>
                <td>Model Dimension</td>
                <td>512</td>
                <td>512</td>
                <td class="success">✅ 一致</td>
            </tr>
            <tr>
                <td>Attention Heads</td>
                <td>8</td>
                <td>8</td>
                <td class="success">✅ 一致</td>
            </tr>
            <tr>
                <td>Beam Size</td>
                <td>3</td>
                <td>3</td>
                <td class="success">✅ 一致</td>
            </tr>
        </table>

        <h2>🚀 后续改进建议</h2>
        
        <div class="warning">
            <h3>短期改进 (立即可行)</h3>
            <ul>
                <li><strong>延长训练时间</strong>: 从5个epoch增加到50-100个epoch</li>
                <li><strong>监控收敛</strong>: 观察loss和BLEU分数的收敛趋势</li>
                <li><strong>验证数据预处理</strong>: 确保与论文完全一致</li>
            </ul>
        </div>

        <div class="success">
            <h3>中期优化 (需要进一步研究)</h3>
            <ul>
                <li><strong>混合精度训练</strong>: 解决BatchNorm兼容性问题，实现FP16/FP8训练</li>
                <li><strong>批次大小优化</strong>: 根据显存情况动态调整</li>
                <li><strong>梯度累积</strong>: 在显存受限时保持有效批次大小</li>
            </ul>
        </div>

        <h2>📊 实验数据</h2>
        
        <h3>训练配置</h3>
        <ul>
            <li><strong>精度</strong>: FP32</li>
            <li><strong>批次大小</strong>: 12</li>
            <li><strong>训练轮数</strong>: 5</li>
            <li><strong>训练时间</strong>: 0.11小时</li>
            <li><strong>数据集</strong>: IU X-Ray (2069训练样本, 296验证样本, 590测试样本)</li>
        </ul>

        <h3>最终指标</h3>
        <ul>
            <li><strong>验证集 BLEU_4</strong>: 0.064</li>
            <li><strong>测试集 BLEU_4</strong>: 0.081</li>
            <li><strong>测试集 METEOR</strong>: 0.119</li>
            <li><strong>测试集 ROUGE_L</strong>: 0.312</li>
        </ul>

        <h2>🎉 结论</h2>
        
        <div class="success">
            <p><strong>本次诊断和修复工作取得了重大突破</strong>:</p>
            <ol>
                <li><strong>成功识别根本原因</strong>: Tokenizer解码bug导致的评估失效</li>
                <li><strong>实施有效修复</strong>: 简单但关键的代码修改解决了核心问题</li>
                <li><strong>验证修复效果</strong>: BLEU分数恢复到合理范围，提升了10^18倍</li>
                <li><strong>建立正确基线</strong>: 为后续的精度对比和优化实验奠定了基础</li>
            </ol>
            
            <p>这次经历证明了<span class="highlight">系统性诊断和代码审查</span>在深度学习项目中的重要性。一个看似微小的bug可能导致整个评估系统失效，而正确的诊断方法能够快速定位并解决问题。</p>
        </div>

        <hr style="margin: 30px 0;">
        <p style="text-align: center; color: #7f8c8d; font-size: 14px;">
            报告生成于 {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')} | R2Gen诊断与修复项目
        </p>
    </div>
</body>
</html>
    """
    
    # 保存HTML文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'R2Gen_Diagnostic_Report_{timestamp}.html'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📊 诊断报告已生成: {filename}")
    return filename

if __name__ == "__main__":
    generate_html_report()

"""
报告质量评估器
对不同精度模型生成的医学报告进行定性和定量评估
"""

import torch
import json
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


class ReportQualityEvaluator:
    """
    报告质量评估器
    
    功能：
    - 对比不同精度模型的生成质量
    - 生成样本报告展示
    - 提供多维度质量分析
    - 生成可视化对比报告
    """
    
    def __init__(self, 
                 models_dict: Dict[str, torch.nn.Module],
                 tokenizer,
                 device: str = 'cuda:0'):
        """
        初始化报告质量评估器
        
        Args:
            models_dict: 不同精度的模型字典 {'fp32': model, 'fp16': model, ...}
            tokenizer: 分词器
            device: 设备
        """
        self.models_dict = models_dict
        self.tokenizer = tokenizer
        self.device = device
        
        # 评估结果存储
        self.evaluation_results = {}
        self.sample_reports = {}
        
        # 质量评估维度
        self.quality_dimensions = {
            'medical_accuracy': '医学准确性',
            'completeness': '完整性',
            'coherence': '连贯性',
            'specificity': '具体性',
            'clinical_relevance': '临床相关性'
        }
        
        print(f"✅ 报告质量评估器初始化完成")
        print(f"   模型数量: {len(models_dict)}")
        print(f"   评估维度: {len(self.quality_dimensions)}")
    
    def generate_sample_reports(self, 
                               test_images: List[torch.Tensor],
                               test_image_ids: List[str],
                               ground_truth_reports: List[str],
                               num_samples: int = 5) -> Dict[str, List[Dict]]:
        """
        为每个模型生成样本报告
        
        Args:
            test_images: 测试图像列表
            test_image_ids: 图像ID列表
            ground_truth_reports: 真实报告列表
            num_samples: 生成样本数量
            
        Returns:
            每个模型的样本报告字典
        """
        print(f"\n🔬 开始生成样本报告 (每个模型{num_samples}个样本)...")
        
        # 选择样本
        sample_indices = np.random.choice(len(test_images), num_samples, replace=False)
        
        results = {}
        
        for model_name, model in self.models_dict.items():
            print(f"\n  📝 生成 {model_name} 模型报告...")
            model.eval()
            
            model_samples = []
            
            with torch.no_grad():
                for i, idx in enumerate(sample_indices):
                    print(f"    样本 {i+1}/{num_samples}: 图像 {test_image_ids[idx]}")
                    
                    # 准备输入
                    image = test_images[idx].unsqueeze(0).to(self.device)
                    
                    # 生成报告
                    start_time = time.time()
                    
                    if model_name == 'fp16':
                        with torch.cuda.amp.autocast():
                            generated_ids = model(image, mode='sample')
                    else:
                        generated_ids = model(image, mode='sample')
                    
                    generation_time = time.time() - start_time
                    
                    # 解码生成的报告
                    if isinstance(generated_ids, list):
                        generated_report = generated_ids[0]  # 取第一个生成结果
                    else:
                        generated_report = self.tokenizer.decode_batch(generated_ids.cpu().numpy())[0]
                    
                    # 清理报告文本
                    generated_report = self._clean_report_text(generated_report)
                    ground_truth = self._clean_report_text(ground_truth_reports[idx])
                    
                    # 计算基础指标
                    metrics = self._calculate_basic_metrics(generated_report, ground_truth)
                    
                    sample_data = {
                        'image_id': test_image_ids[idx],
                        'sample_index': i + 1,
                        'generated_report': generated_report,
                        'ground_truth_report': ground_truth,
                        'generation_time_ms': generation_time * 1000,
                        'metrics': metrics,
                        'model_precision': model_name
                    }
                    
                    model_samples.append(sample_data)
                    
                    print(f"      生成时间: {generation_time*1000:.1f}ms")
                    print(f"      BLEU-4: {metrics['bleu4']:.3f}")
            
            results[model_name] = model_samples
            print(f"  ✅ {model_name} 模型报告生成完成")
        
        self.sample_reports = results
        return results
    
    def _clean_report_text(self, text: str) -> str:
        """清理报告文本"""
        if isinstance(text, list):
            text = ' '.join(text)
        
        # 移除特殊标记
        text = text.replace('<unk>', '').replace('<pad>', '').replace('<eos>', '')
        text = text.replace('<bos>', '').replace('<start>', '').replace('<end>', '')
        
        # 清理多余空格
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _calculate_basic_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        """计算基础指标"""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        
        try:
            # 分词
            gen_tokens = word_tokenize(generated.lower())
            ref_tokens = word_tokenize(reference.lower())
            
            # BLEU分数
            smoothie = SmoothingFunction().method4
            bleu1 = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
            bleu4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
            
            # 长度比较
            length_ratio = len(gen_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
            
            # 词汇重叠
            gen_set = set(gen_tokens)
            ref_set = set(ref_tokens)
            vocab_overlap = len(gen_set & ref_set) / len(ref_set) if len(ref_set) > 0 else 0
            
            return {
                'bleu1': bleu1,
                'bleu4': bleu4,
                'length_ratio': length_ratio,
                'vocab_overlap': vocab_overlap,
                'generated_length': len(gen_tokens),
                'reference_length': len(ref_tokens)
            }
        except Exception as e:
            print(f"⚠️ 指标计算错误: {e}")
            return {
                'bleu1': 0.0, 'bleu4': 0.0, 'length_ratio': 0.0, 
                'vocab_overlap': 0.0, 'generated_length': 0, 'reference_length': 0
            }
    
    def analyze_report_quality(self) -> Dict[str, Any]:
        """
        分析报告质量
        
        Returns:
            质量分析结果
        """
        if not self.sample_reports:
            raise ValueError("请先生成样本报告")
        
        print(f"\n📊 开始报告质量分析...")
        
        analysis_results = {}
        
        for model_name, samples in self.sample_reports.items():
            print(f"\n  🔍 分析 {model_name} 模型...")
            
            # 收集指标
            metrics_list = [sample['metrics'] for sample in samples]
            generation_times = [sample['generation_time_ms'] for sample in samples]
            
            # 计算统计信息
            model_analysis = {
                'model_name': model_name,
                'num_samples': len(samples),
                'avg_bleu1': np.mean([m['bleu1'] for m in metrics_list]),
                'avg_bleu4': np.mean([m['bleu4'] for m in metrics_list]),
                'avg_length_ratio': np.mean([m['length_ratio'] for m in metrics_list]),
                'avg_vocab_overlap': np.mean([m['vocab_overlap'] for m in metrics_list]),
                'avg_generation_time_ms': np.mean(generation_times),
                'std_generation_time_ms': np.std(generation_times),
                'avg_generated_length': np.mean([m['generated_length'] for m in metrics_list]),
                'avg_reference_length': np.mean([m['reference_length'] for m in metrics_list])
            }
            
            # 质量评分 (综合指标)
            quality_score = (
                model_analysis['avg_bleu4'] * 0.4 +
                model_analysis['avg_vocab_overlap'] * 0.3 +
                min(1.0, model_analysis['avg_length_ratio']) * 0.3
            )
            model_analysis['quality_score'] = quality_score
            
            analysis_results[model_name] = model_analysis
            
            print(f"    BLEU-4: {model_analysis['avg_bleu4']:.3f}")
            print(f"    词汇重叠: {model_analysis['avg_vocab_overlap']:.3f}")
            print(f"    长度比例: {model_analysis['avg_length_ratio']:.3f}")
            print(f"    质量评分: {quality_score:.3f}")
            print(f"    生成速度: {model_analysis['avg_generation_time_ms']:.1f}ms")
        
        self.evaluation_results = analysis_results
        return analysis_results
    
    def generate_comparison_report(self, save_path: str = "report_quality_comparison.html") -> str:
        """
        生成详细的对比报告
        
        Args:
            save_path: 保存路径
            
        Returns:
            生成的HTML报告路径
        """
        if not self.sample_reports or not self.evaluation_results:
            raise ValueError("请先生成样本报告和质量分析")
        
        print(f"\n📄 生成对比报告...")
        
        html_content = self._create_html_report()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ 对比报告已保存: {save_path}")
        return save_path
    
    def _create_html_report(self) -> str:
        """创建HTML报告"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        model_names = ', '.join(self.sample_reports.keys())
        num_samples = len(list(self.sample_reports.values())[0])

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>R2Gen模型报告质量对比</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .model-section {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }}
        .fp32 {{ border-left: 5px solid #3498db; }}
        .fp16 {{ border-left: 5px solid #e74c3c; }}
        .fp8 {{ border-left: 5px solid #2ecc71; }}
        .sample {{ background: #f9f9f9; margin: 10px 0; padding: 15px; border-radius: 3px; }}
        .metrics {{ display: flex; gap: 20px; margin: 10px 0; }}
        .metric {{ background: white; padding: 10px; border-radius: 3px; text-align: center; }}
        .generated {{ background: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .reference {{ background: #f0f8ff; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .comparison-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .comparison-table th, .comparison-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .comparison-table th {{ background: #f4f4f4; }}
        .best {{ background: #d4edda; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 R2Gen模型报告质量对比分析</h1>
        <p>生成时间: {timestamp}</p>
        <p>对比模型: {model_names}</p>
        <p>样本数量: {num_samples} 个/模型</p>
    </div>
"""
        
        # 添加总体对比表格
        html += self._create_summary_table()
        
        # 添加详细样本对比
        html += self._create_detailed_samples()
        
        html += """
</body>
</html>
"""
        return html
    
    def _create_summary_table(self) -> str:
        """创建总体对比表格"""
        html = """
    <h2>📊 总体性能对比</h2>
    <table class="comparison-table">
        <tr>
            <th>模型</th>
            <th>BLEU-1</th>
            <th>BLEU-4</th>
            <th>词汇重叠</th>
            <th>长度比例</th>
            <th>质量评分</th>
            <th>生成速度(ms)</th>
        </tr>
"""
        
        # 找到最佳值用于高亮
        best_bleu4 = max(result['avg_bleu4'] for result in self.evaluation_results.values())
        best_quality = max(result['quality_score'] for result in self.evaluation_results.values())
        best_speed = min(result['avg_generation_time_ms'] for result in self.evaluation_results.values())
        
        for model_name, result in self.evaluation_results.items():
            bleu4_class = 'best' if result['avg_bleu4'] == best_bleu4 else ''
            quality_class = 'best' if result['quality_score'] == best_quality else ''
            speed_class = 'best' if result['avg_generation_time_ms'] == best_speed else ''
            
            html += f"""
        <tr>
            <td><strong>{model_name.upper()}</strong></td>
            <td>{result['avg_bleu1']:.3f}</td>
            <td class="{bleu4_class}">{result['avg_bleu4']:.3f}</td>
            <td>{result['avg_vocab_overlap']:.3f}</td>
            <td>{result['avg_length_ratio']:.3f}</td>
            <td class="{quality_class}">{result['quality_score']:.3f}</td>
            <td class="{speed_class}">{result['avg_generation_time_ms']:.1f}</td>
        </tr>
"""
        
        html += """
    </table>
"""
        return html
    
    def _create_detailed_samples(self) -> str:
        """创建详细样本对比"""
        html = "<h2>📝 详细样本对比</h2>\n"
        
        # 按样本索引组织数据
        num_samples = len(list(self.sample_reports.values())[0])
        
        for sample_idx in range(num_samples):
            html += f"<h3>样本 {sample_idx + 1}</h3>\n"
            
            # 显示真实报告
            reference_report = list(self.sample_reports.values())[0][sample_idx]['ground_truth_report']
            html += f'<div class="reference"><strong>真实报告:</strong><br>{reference_report}</div>\n'
            
            # 显示每个模型的生成结果
            for model_name in self.sample_reports.keys():
                sample = self.sample_reports[model_name][sample_idx]
                
                html += f"""
<div class="model-section {model_name}">
    <h4>{model_name.upper()} 模型生成</h4>
    <div class="generated"><strong>生成报告:</strong><br>{sample['generated_report']}</div>
    <div class="metrics">
        <div class="metric">
            <strong>BLEU-4</strong><br>
            {sample['metrics']['bleu4']:.3f}
        </div>
        <div class="metric">
            <strong>词汇重叠</strong><br>
            {sample['metrics']['vocab_overlap']:.3f}
        </div>
        <div class="metric">
            <strong>长度比例</strong><br>
            {sample['metrics']['length_ratio']:.3f}
        </div>
        <div class="metric">
            <strong>生成时间</strong><br>
            {sample['generation_time_ms']:.1f}ms
        </div>
    </div>
</div>
"""
        
        return html
    
    def print_summary(self):
        """打印评估摘要"""
        if not self.evaluation_results:
            print("❌ 请先运行质量分析")
            return
        
        print(f"\n📊 报告质量评估摘要")
        print(f"=" * 60)
        
        # 按质量评分排序
        sorted_results = sorted(
            self.evaluation_results.items(),
            key=lambda x: x[1]['quality_score'],
            reverse=True
        )
        
        print(f"{'模型':<8} {'质量评分':<10} {'BLEU-4':<10} {'生成速度':<12} {'综合排名'}")
        print(f"{'-'*60}")
        
        for rank, (model_name, result) in enumerate(sorted_results, 1):
            print(f"{model_name.upper():<8} "
                  f"{result['quality_score']:<10.3f} "
                  f"{result['avg_bleu4']:<10.3f} "
                  f"{result['avg_generation_time_ms']:<12.1f} "
                  f"#{rank}")
        
        # 最佳模型推荐
        best_model = sorted_results[0][0]
        print(f"\n🏆 推荐模型: {best_model.upper()}")
        print(f"   理由: 在质量评分中表现最佳")
        
        # 速度最快模型
        fastest_model = min(
            self.evaluation_results.items(),
            key=lambda x: x[1]['avg_generation_time_ms']
        )[0]
        print(f"⚡ 速度最快: {fastest_model.upper()}")


if __name__ == "__main__":
    print("🧪 报告质量评估器测试...")
    print("注意: 这是一个框架测试，需要实际的模型和数据来运行完整评估")
    
    # 这里只是展示使用方法
    print("""
使用示例:
    # 1. 准备不同精度的模型
    models = {
        'fp32': model_fp32,
        'fp16': model_fp16,
        'fp8': model_fp8
    }
    
    # 2. 创建评估器
    evaluator = ReportQualityEvaluator(models, tokenizer)
    
    # 3. 生成样本报告
    evaluator.generate_sample_reports(test_images, image_ids, ground_truth)
    
    # 4. 分析质量
    evaluator.analyze_report_quality()
    
    # 5. 生成对比报告
    evaluator.generate_comparison_report()
    
    # 6. 打印摘要
    evaluator.print_summary()
    """)

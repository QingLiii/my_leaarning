#!/usr/bin/env python3
"""
测试报告质量评估器
"""

import sys
import torch
import numpy as np
from datetime import datetime

# 添加路径
sys.path.append('R2Gen-main')

from modules.report_quality_evaluator import ReportQualityEvaluator


class MockModel(torch.nn.Module):
    """模拟模型用于测试"""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.linear = torch.nn.Linear(100, 1000)  # 简单的线性层
        
        # 不同模型的报告模板（模拟不同质量）
        self.report_templates = {
            'fp32': [
                "The chest X-ray shows clear lungs with no acute cardiopulmonary abnormalities. Heart size is normal.",
                "Bilateral lower lobe consolidation consistent with pneumonia. Recommend clinical correlation.",
                "No acute findings. Normal cardiac silhouette and clear lung fields.",
                "Small pleural effusion on the right side. Otherwise unremarkable.",
                "Mild cardiomegaly noted. No acute pulmonary findings."
            ],
            'fp16': [
                "Chest X-ray demonstrates clear lungs. No acute abnormalities. Normal heart size.",
                "Lower lobe consolidation bilaterally. Findings suggest pneumonia.",
                "Normal cardiac and pulmonary findings. No acute disease.",
                "Right pleural effusion present. Otherwise normal.",
                "Enlarged heart noted. Lungs are clear."
            ],
            'fp8': [
                "Clear lungs. Normal heart.",
                "Consolidation both lungs. Pneumonia likely.",
                "Normal chest.",
                "Right effusion.",
                "Big heart. Clear lungs."
            ]
        }
    
    def forward(self, images, reports_ids=None, mode='sample'):
        batch_size = images.size(0)
        
        if mode == 'sample':
            # 模拟生成报告（返回预定义的报告）
            reports = []
            for i in range(batch_size):
                # 根据模型类型选择不同质量的报告
                template_idx = i % len(self.report_templates[self.model_name])
                report = self.report_templates[self.model_name][template_idx]
                reports.append(report)
            return reports
        else:
            # 训练模式
            return torch.randn(batch_size, 60, 1000, device=images.device)


class MockTokenizer:
    """模拟分词器"""
    
    def decode_batch(self, token_ids):
        """模拟解码"""
        # 返回预定义的真实报告
        ground_truth_reports = [
            "The chest radiograph demonstrates clear bilateral lung fields with no evidence of acute cardiopulmonary abnormalities. The cardiac silhouette appears normal in size and configuration.",
            "Bilateral lower lobe airspace consolidation is present, consistent with pneumonia. Clinical correlation is recommended for appropriate antibiotic therapy.",
            "No acute cardiopulmonary findings are identified. The heart size is within normal limits and the lung fields are clear.",
            "A small right-sided pleural effusion is noted. Otherwise, the examination is unremarkable with normal cardiac and pulmonary findings.",
            "Mild cardiomegaly is observed. No acute pulmonary abnormalities are detected. The lung fields appear clear."
        ]
        
        results = []
        for i, token_id in enumerate(token_ids):
            idx = i % len(ground_truth_reports)
            results.append(ground_truth_reports[idx])
        
        return results


def test_quality_evaluator():
    """测试报告质量评估器"""
    print("🧪 测试报告质量评估器...")
    
    # 创建模拟模型
    models = {
        'fp32': MockModel('fp32'),
        'fp16': MockModel('fp16'),
        'fp8': MockModel('fp8')
    }
    
    # 创建模拟分词器
    tokenizer = MockTokenizer()
    
    # 创建评估器
    evaluator = ReportQualityEvaluator(models, tokenizer)
    
    # 创建测试数据
    num_samples = 5
    test_images = [torch.randn(3, 224, 224) for _ in range(num_samples)]
    test_image_ids = [f"test_image_{i+1}" for i in range(num_samples)]
    ground_truth_reports = [
        "The chest radiograph demonstrates clear bilateral lung fields with no evidence of acute cardiopulmonary abnormalities. The cardiac silhouette appears normal in size and configuration.",
        "Bilateral lower lobe airspace consolidation is present, consistent with pneumonia. Clinical correlation is recommended for appropriate antibiotic therapy.",
        "No acute cardiopulmonary findings are identified. The heart size is within normal limits and the lung fields are clear.",
        "A small right-sided pleural effusion is noted. Otherwise, the examination is unremarkable with normal cardiac and pulmonary findings.",
        "Mild cardiomegaly is observed. No acute pulmonary abnormalities are detected. The lung fields appear clear."
    ]
    
    print(f"📊 测试数据准备完成:")
    print(f"   样本数量: {num_samples}")
    print(f"   模型数量: {len(models)}")
    
    # 生成样本报告
    print(f"\n🔬 生成样本报告...")
    sample_reports = evaluator.generate_sample_reports(
        test_images=test_images,
        test_image_ids=test_image_ids,
        ground_truth_reports=ground_truth_reports,
        num_samples=num_samples
    )
    
    # 分析质量
    print(f"\n📊 分析报告质量...")
    quality_analysis = evaluator.analyze_report_quality()
    
    # 生成对比报告
    print(f"\n📄 生成对比报告...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"test_quality_report_{timestamp}.html"
    evaluator.generate_comparison_report(report_path)
    
    # 打印摘要
    print(f"\n📋 评估摘要:")
    evaluator.print_summary()
    
    # 显示详细结果
    print(f"\n📝 详细样本对比:")
    for i in range(num_samples):
        print(f"\n样本 {i+1} ({test_image_ids[i]}):")
        print(f"真实报告: {ground_truth_reports[i][:100]}...")
        
        for model_name in models.keys():
            sample = sample_reports[model_name][i]
            print(f"{model_name.upper()}: {sample['generated_report'][:100]}...")
            print(f"  BLEU-4: {sample['metrics']['bleu4']:.3f}, "
                  f"词汇重叠: {sample['metrics']['vocab_overlap']:.3f}")
    
    print(f"\n✅ 测试完成!")
    print(f"📄 详细报告已保存: {report_path}")
    
    return evaluator, sample_reports, quality_analysis


def demonstrate_usage():
    """演示使用方法"""
    print(f"\n💡 报告质量评估器使用方法:")
    print("""
1. 准备不同精度的训练好的模型:
   models = {
       'fp32': trained_model_fp32,
       'fp16': trained_model_fp16,
       'fp8': trained_model_fp8
   }

2. 创建评估器:
   evaluator = ReportQualityEvaluator(models, tokenizer)

3. 生成样本报告:
   sample_reports = evaluator.generate_sample_reports(
       test_images, image_ids, ground_truth_reports, num_samples=5
   )

4. 分析质量:
   quality_analysis = evaluator.analyze_report_quality()

5. 生成HTML对比报告:
   evaluator.generate_comparison_report('quality_report.html')

6. 查看摘要:
   evaluator.print_summary()

输出包括:
- 定量指标对比 (BLEU, 词汇重叠等)
- 每个样本的详细对比
- 质量评分和排名
- 生成速度分析
- 可视化HTML报告
""")


if __name__ == "__main__":
    print("🚀 报告质量评估器测试")
    print("=" * 50)
    
    # 运行测试
    evaluator, sample_reports, quality_analysis = test_quality_evaluator()
    
    # 演示使用方法
    demonstrate_usage()
    
    print(f"\n🎯 测试结论:")
    print(f"✅ 报告质量评估器功能正常")
    print(f"✅ 可以对比不同精度模型的生成质量")
    print(f"✅ 提供定量和定性的评估指标")
    print(f"✅ 生成专业的HTML对比报告")
    print(f"✅ 适用于R2Gen精度对比实验")

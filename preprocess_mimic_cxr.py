#!/usr/bin/env python3
"""
MIMIC-CXR数据预处理脚本
基于R2Gen的数据格式要求，将MIMIC-CXR原始数据转换为训练所需的格式
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from pathlib import Path
import argparse
from tqdm import tqdm

def clean_report_mimic_cxr(report):
    """
    清理MIMIC-CXR报告文本
    基于R2Gen的清理逻辑
    """
    if not report or pd.isna(report):
        return ""
    
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\\[\\]{}]', '', 
                                   t.replace('"', '').replace('/', '').replace('\\\\', '').replace("'", '').strip().lower())
    
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != '']
    report = ' . '.join(tokens) + ' .'
    
    return report

def load_mimic_cxr_reports(reports_dir):
    """
    加载MIMIC-CXR报告数据
    """
    print("📄 加载MIMIC-CXR报告数据...")
    
    reports_data = {}
    reports_path = Path(reports_dir)
    
    # 查找所有报告文件
    report_files = list(reports_path.rglob("*.txt"))
    
    print(f"找到 {len(report_files)} 个报告文件")
    
    for report_file in tqdm(report_files, desc="加载报告"):
        try:
            # 从文件路径提取study_id
            # 路径格式: p10/p10000032/s50414267.txt
            parts = report_file.parts
            if len(parts) >= 3:
                study_id = parts[-1].replace('.txt', '')  # s50414267
                
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析报告内容
                sections = {}
                current_section = None
                current_content = []
                
                for line in content.split('\n'):
                    line = line.strip()
                    if line.endswith(':') and len(line) < 50:  # 可能是章节标题
                        if current_section:
                            sections[current_section] = '\n'.join(current_content)
                        current_section = line[:-1].upper()
                        current_content = []
                    else:
                        current_content.append(line)
                
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                
                # 提取FINDINGS和IMPRESSION部分
                findings = sections.get('FINDINGS', '')
                impression = sections.get('IMPRESSION', '')
                
                # 合并为完整报告
                full_report = ''
                if findings:
                    full_report += findings
                if impression:
                    if full_report:
                        full_report += ' '
                    full_report += impression
                
                if full_report:
                    reports_data[study_id] = clean_report_mimic_cxr(full_report)
                    
        except Exception as e:
            print(f"处理报告文件 {report_file} 时出错: {e}")
            continue
    
    print(f"✅ 成功加载 {len(reports_data)} 个报告")
    return reports_data

def load_mimic_cxr_metadata(metadata_path):
    """
    加载MIMIC-CXR元数据
    """
    print("📊 加载MIMIC-CXR元数据...")
    
    try:
        metadata = pd.read_csv(metadata_path)
        print(f"✅ 元数据包含 {len(metadata)} 条记录")
        print(f"列名: {list(metadata.columns)}")
        return metadata
    except Exception as e:
        print(f"❌ 加载元数据失败: {e}")
        return None

def find_image_files(images_dir):
    """
    查找所有图像文件
    """
    print("🖼️ 查找图像文件...")
    
    image_files = {}
    images_path = Path(images_dir)
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.dcm']
    
    for ext in image_extensions:
        files = list(images_path.rglob(f"*{ext}"))
        print(f"找到 {len(files)} 个 {ext} 文件")
        
        for img_file in files:
            # 从路径提取study_id和dicom_id
            # 路径格式: p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
            parts = img_file.parts
            if len(parts) >= 4:
                study_id = parts[-2]  # s50414267
                dicom_id = parts[-1].replace(ext, '')  # 02aa804e-bde0afdd-112c0b34-7bc16630-4e384014
                
                if study_id not in image_files:
                    image_files[study_id] = []
                
                # 存储相对路径
                rel_path = str(img_file.relative_to(images_path))
                image_files[study_id].append(rel_path)
    
    print(f"✅ 找到 {len(image_files)} 个study的图像文件")
    return image_files

def create_train_val_test_split(data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    创建训练/验证/测试集划分
    """
    print("📊 创建数据集划分...")
    
    study_ids = list(data.keys())
    np.random.seed(42)  # 确保可重复性
    np.random.shuffle(study_ids)
    
    n_total = len(study_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = study_ids[:n_train]
    val_ids = study_ids[n_train:n_train + n_val]
    test_ids = study_ids[n_train + n_val:]
    
    print(f"训练集: {len(train_ids)} 样本")
    print(f"验证集: {len(val_ids)} 样本")
    print(f"测试集: {len(test_ids)} 样本")
    
    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

def create_annotation_json(reports_data, image_files, splits, output_path):
    """
    创建R2Gen格式的annotation.json文件
    """
    print("📝 创建annotation.json文件...")
    
    annotation = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for split_name, study_ids in splits.items():
        print(f"处理 {split_name} 集...")
        
        for study_id in tqdm(study_ids, desc=f"创建{split_name}集"):
            if study_id in reports_data and study_id in image_files:
                report = reports_data[study_id]
                images = image_files[study_id]
                
                # 过滤掉空报告
                if len(report.strip()) > 10:  # 至少10个字符
                    # 对于MIMIC-CXR，通常每个study只取一张图像
                    # 如果有多张图像，取第一张
                    image_path = images[0] if images else None
                    
                    if image_path:
                        annotation[split_name].append({
                            'id': study_id,
                            'image_path': [image_path],  # R2Gen格式要求列表
                            'report': report
                        })
    
    # 统计信息
    for split_name in ['train', 'val', 'test']:
        print(f"{split_name}集: {len(annotation[split_name])} 个样本")
    
    # 保存annotation.json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)
    
    print(f"✅ annotation.json已保存到: {output_path}")
    return annotation

def main():
    parser = argparse.ArgumentParser(description='MIMIC-CXR数据预处理')
    parser.add_argument('--data_dir', type=str, default='datasets/mimic-cxr-dataset',
                        help='MIMIC-CXR数据目录')
    parser.add_argument('--output_dir', type=str, default='R2Gen-main/data/mimic_cxr',
                        help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数（用于测试）')
    
    args = parser.parse_args()
    
    print("🚀 开始MIMIC-CXR数据预处理...")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载报告数据
    reports_dir = os.path.join(args.data_dir, 'mimic-cxr-reports/files')
    if not os.path.exists(reports_dir):
        print(f"❌ 报告目录不存在: {reports_dir}")
        return
    
    reports_data = load_mimic_cxr_reports(reports_dir)
    
    if not reports_data:
        print("❌ 没有找到有效的报告数据")
        return
    
    # 2. 查找图像文件
    images_dir = os.path.join(args.data_dir, 'official_data_iccv_final/files')
    if not os.path.exists(images_dir):
        print(f"❌ 图像目录不存在: {images_dir}")
        return
    
    image_files = find_image_files(images_dir)
    
    if not image_files:
        print("❌ 没有找到有效的图像文件")
        return
    
    # 3. 匹配报告和图像
    matched_data = {}
    for study_id in reports_data:
        if study_id in image_files:
            matched_data[study_id] = {
                'report': reports_data[study_id],
                'images': image_files[study_id]
            }
    
    print(f"✅ 匹配到 {len(matched_data)} 个有效样本")
    
    # 限制样本数量（用于测试）
    if args.max_samples and len(matched_data) > args.max_samples:
        study_ids = list(matched_data.keys())[:args.max_samples]
        matched_data = {sid: matched_data[sid] for sid in study_ids}
        print(f"🔄 限制样本数量为: {len(matched_data)}")
    
    # 4. 创建数据集划分
    splits = create_train_val_test_split(matched_data)
    
    # 5. 创建annotation.json
    annotation_path = os.path.join(args.output_dir, 'annotation.json')
    annotation = create_annotation_json(reports_data, image_files, splits, annotation_path)
    
    # 6. 创建图像软链接
    images_output_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    
    print("🔗 创建图像软链接...")
    images_source_dir = os.path.abspath(images_dir)
    images_target_dir = os.path.abspath(images_output_dir)
    
    # 创建软链接指向原始图像目录
    link_path = os.path.join(images_target_dir, 'files')
    if not os.path.exists(link_path):
        try:
            os.symlink(images_source_dir, link_path)
            print(f"✅ 创建软链接: {link_path} -> {images_source_dir}")
        except Exception as e:
            print(f"⚠️ 创建软链接失败: {e}")
            print("将复制图像路径信息到配置中")
    
    print("🎉 MIMIC-CXR数据预处理完成！")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📄 annotation.json: {annotation_path}")
    print(f"🖼️ 图像目录: {images_output_dir}")

if __name__ == "__main__":
    main()

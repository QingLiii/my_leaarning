#!/usr/bin/env python3
"""
修复MIMIC-CXR annotation.json中的图像路径
"""

import json
import os

def fix_image_paths():
    """修复图像路径，添加files/前缀"""
    
    annotation_path = 'R2Gen-main/data/mimic_cxr/annotation.json'
    
    print("🔧 修复MIMIC-CXR图像路径...")
    
    # 读取原始annotation
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    # 修复每个split的路径
    for split_name in ['train', 'val', 'test']:
        print(f"修复 {split_name} 集...")
        
        for item in annotation[split_name]:
            # 为每个图像路径添加files/前缀
            new_image_paths = []
            for img_path in item['image_path']:
                if not img_path.startswith('files/'):
                    new_path = f"files/{img_path}"
                    new_image_paths.append(new_path)
                else:
                    new_image_paths.append(img_path)
            
            item['image_path'] = new_image_paths
    
    # 保存修复后的annotation
    with open(annotation_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 路径修复完成，已保存到: {annotation_path}")
    
    # 验证修复结果
    print("🔍 验证修复结果...")
    sample_item = annotation['train'][0]
    sample_path = sample_item['image_path'][0]
    full_path = os.path.join('R2Gen-main/data/mimic_cxr/images', sample_path)
    
    if os.path.exists(full_path):
        print(f"✅ 路径验证成功: {sample_path}")
    else:
        print(f"❌ 路径验证失败: {sample_path}")
        print(f"完整路径: {full_path}")

if __name__ == "__main__":
    fix_image_paths()

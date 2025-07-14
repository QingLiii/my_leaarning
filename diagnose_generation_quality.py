#!/usr/bin/env python3
"""
R2Gen生成质量诊断脚本
检查模型生成的文本质量，诊断BLEU分数异常低的原因
"""

import sys
import torch
import numpy as np
sys.path.append('R2Gen-main')

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from models.r2gen import R2GenModel
import argparse

def create_args():
    """创建参数配置"""
    args = argparse.Namespace()
    
    # 数据集相关
    args.image_dir = 'datasets/iu_xray/images/'
    args.ann_path = 'datasets/iu_xray/annotation.json'
    args.dataset_name = 'iu_xray'
    args.max_seq_length = 60
    args.threshold = 3
    args.num_workers = 2
    args.batch_size = 4  # 小批次用于诊断
    
    # 模型相关
    args.d_model = 512
    args.d_ff = 512
    args.d_vf = 2048
    args.num_heads = 8
    args.num_layers = 3
    args.dropout = 0.1
    args.logit_layers = 1
    args.bos_idx = 0
    args.eos_idx = 0
    args.drop_prob_lm = 0.5
    
    # 记忆模块
    args.rm_num_slots = 3
    args.rm_num_heads = 8
    args.rm_d_model = 512
    
    # 视觉特征提取
    args.visual_extractor = 'resnet101'
    args.visual_extractor_pretrained = True
    
    # 采样相关
    args.sample_method = 'beam_search'
    args.beam_size = 3
    args.temperature = 1.0
    args.sample_n = 1
    
    return args

def load_model(model_path, args, tokenizer):
    """加载训练好的模型"""
    print(f"📥 加载模型: {model_path}")
    model = R2GenModel(args, tokenizer)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    model.cuda()
    model.eval()
    return model

def diagnose_generation(model, dataloader, tokenizer, num_samples=5):
    """诊断生成质量"""
    print(f"\n🔍 开始诊断生成质量 (检查{num_samples}个样本)...")
    
    model.eval()
    with torch.no_grad():
        for i, (images_id, images, reports_ids, reports_masks) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            print(f"\n--- 样本 {i+1} ---")
            
            # 移动到GPU
            images = images.cuda()
            reports_ids = reports_ids.cuda()
            
            # 生成报告
            try:
                generated_ids = model(images, mode='sample')
                print(f"✅ 生成成功")
                
                # 检查生成的ID
                if isinstance(generated_ids, torch.Tensor):
                    gen_ids_cpu = generated_ids.cpu().numpy()
                    print(f"📊 生成ID形状: {gen_ids_cpu.shape}")
                    print(f"📊 生成ID范围: [{gen_ids_cpu.min()}, {gen_ids_cpu.max()}]")
                    print(f"📊 生成ID样例: {gen_ids_cpu[0][:10]}")  # 前10个token
                    
                    # 解码生成的文本
                    if len(gen_ids_cpu.shape) == 2:
                        generated_text = tokenizer.decode(gen_ids_cpu[0])
                    else:
                        generated_text = "解码错误：维度不匹配"
                else:
                    generated_text = str(generated_ids)
                    print(f"⚠️ 生成结果不是tensor: {type(generated_ids)}")
                
                print(f"🤖 生成文本: '{generated_text}'")
                print(f"📏 生成文本长度: {len(generated_text)}")
                
                # 解码参考文本
                ref_ids_cpu = reports_ids.cpu().numpy()
                ref_text = tokenizer.decode(ref_ids_cpu[0])
                print(f"📖 参考文本: '{ref_text}'")
                print(f"📏 参考文本长度: {len(ref_text)}")
                
                # 分析问题
                if len(generated_text.strip()) == 0:
                    print("❌ 问题: 生成文本为空！")
                elif len(generated_text.strip()) < 5:
                    print("⚠️ 问题: 生成文本过短！")
                elif generated_text == ref_text:
                    print("🎯 完美匹配（不太可能）")
                else:
                    print("✅ 生成文本正常")
                    
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                import traceback
                traceback.print_exc()

def check_tokenizer(tokenizer):
    """检查tokenizer状态"""
    print(f"\n🔤 检查Tokenizer状态...")
    print(f"📊 词汇表大小: {tokenizer.get_vocab_size()}")
    print(f"📊 token2idx样例: {list(tokenizer.token2idx.items())[:10]}")
    print(f"📊 idx2token样例: {list(tokenizer.idx2token.items())[:10]}")
    
    # 测试编码解码
    test_text = "the lungs are clear"
    encoded = tokenizer(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"🧪 测试编码: '{test_text}' -> {encoded}")
    print(f"🧪 测试解码: {encoded} -> '{decoded}'")

def main():
    print("🚀 R2Gen生成质量诊断开始...")
    
    # 创建参数
    args = create_args()
    
    # 创建tokenizer
    print("📝 创建tokenizer...")
    tokenizer = Tokenizer(args)
    check_tokenizer(tokenizer)
    
    # 创建数据加载器
    print("📊 创建数据加载器...")
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=False)
    
    # 检查可用的模型
    import os
    model_paths = [
        'results/precision_comparison/fp32/model_best.pth',
        'results/precision_comparison/fp16/model_best.pth', 
        'results/precision_comparison/fp8/model_best.pth'
    ]
    
    available_models = [path for path in model_paths if os.path.exists(path)]
    
    if not available_models:
        print("❌ 没有找到训练好的模型文件")
        print("请先运行训练实验")
        return
    
    # 诊断第一个可用模型
    model_path = available_models[0]
    precision = model_path.split('/')[-2]
    print(f"🎯 诊断模型: {precision}")
    
    # 加载模型
    model = load_model(model_path, args, tokenizer)
    if model is None:
        return
    
    # 诊断生成质量
    diagnose_generation(model, train_dataloader, tokenizer, num_samples=3)
    
    print(f"\n📋 诊断完成！")
    print(f"如果生成文本为空或异常短，这就是BLEU分数异常低的原因。")

if __name__ == "__main__":
    main()

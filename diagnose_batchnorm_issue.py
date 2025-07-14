#!/usr/bin/env python3
"""
诊断BatchNorm维度不匹配问题
定位具体的错误触发位置和原因
"""

import sys
import torch
import torch.nn as nn
import traceback
from contextlib import contextmanager

sys.path.append('R2Gen-main')

@contextmanager
def debug_autocast(enabled=True, dtype=torch.float16):
    """调试版本的autocast，捕获详细错误信息"""
    if enabled:
        print(f"🔍 开始autocast调试 (dtype={dtype})")
        try:
            with torch.autocast('cuda', dtype=dtype):
                yield
        except Exception as e:
            print(f"❌ autocast错误: {e}")
            print(f"错误类型: {type(e)}")
            traceback.print_exc()
            raise
    else:
        yield

def test_visual_extractor():
    """测试视觉特征提取器的混合精度兼容性"""
    print("\n🔍 测试Visual Extractor...")
    
    try:
        from modules.visual_extractor import VisualExtractor
        import argparse
        
        # 创建测试参数
        args = argparse.Namespace()
        args.visual_extractor = 'resnet101'
        args.visual_extractor_pretrained = True
        args.d_vf = 2048
        
        # 创建模型
        model = VisualExtractor(args).cuda()
        print(f"✅ Visual Extractor创建成功")
        
        # 测试FP32
        print("🧪 测试FP32...")
        x = torch.randn(2, 3, 224, 224).cuda()
        with torch.no_grad():
            att_feats, fc_feats = model(x)
            print(f"✅ FP32成功: att_feats={att_feats.shape}, fc_feats={fc_feats.shape}")
        
        # 测试FP16
        print("🧪 测试FP16...")
        try:
            with debug_autocast(True, torch.float16):
                with torch.no_grad():
                    att_feats, fc_feats = model(x)
                    print(f"✅ FP16成功: att_feats={att_feats.shape}, fc_feats={fc_feats.shape}")
        except Exception as e:
            print(f"❌ FP16失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Visual Extractor测试失败: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_encoder_decoder():
    """测试编码器解码器的混合精度兼容性"""
    print("\n🔍 测试Encoder Decoder...")
    
    try:
        from modules.encoder_decoder import EncoderDecoder
        from modules.tokenizers import Tokenizer
        import argparse
        
        # 创建测试参数
        args = argparse.Namespace()
        args.ann_path = 'datasets/iu_xray/annotation.json'
        args.threshold = 3
        args.dataset_name = 'iu_xray'
        args.d_model = 512
        args.d_ff = 512
        args.d_vf = 2048
        args.num_heads = 8
        args.num_layers = 3
        args.dropout = 0.1
        args.logit_layers = 1
        args.bos_idx = 0
        args.eos_idx = 0
        args.pad_idx = 0
        args.use_bn = True
        args.rm_num_slots = 3
        args.rm_num_heads = 8
        args.rm_d_model = 512
        args.sample_method = 'beam_search'
        args.beam_size = 3
        args.temperature = 1.0
        args.sample_n = 1
        args.drop_prob_lm = 0.5
        
        # 创建tokenizer和模型
        tokenizer = Tokenizer(args)
        model = EncoderDecoder(args, tokenizer).cuda()
        print(f"✅ Encoder Decoder创建成功")
        
        # 创建测试数据
        batch_size = 2
        fc_feats = torch.randn(batch_size, 2048).cuda()
        att_feats = torch.randn(batch_size, 49, 2048).cuda()
        targets = torch.randint(1, 100, (batch_size, 20)).cuda()
        
        # 测试FP32
        print("🧪 测试FP32...")
        with torch.no_grad():
            output = model(fc_feats, att_feats, targets, mode='forward')
            print(f"✅ FP32成功: output shape={output.shape}")
        
        # 测试FP16
        print("🧪 测试FP16...")
        try:
            with debug_autocast(True, torch.float16):
                with torch.no_grad():
                    output = model(fc_feats, att_feats, targets, mode='forward')
                    print(f"✅ FP16成功: output shape={output.shape}")
        except Exception as e:
            print(f"❌ FP16失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Encoder Decoder测试失败: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_full_model():
    """测试完整R2Gen模型的混合精度兼容性"""
    print("\n🔍 测试完整R2Gen模型...")
    
    try:
        from models.r2gen import R2GenModel
        from modules.tokenizers import Tokenizer
        import argparse
        
        # 创建测试参数
        args = argparse.Namespace()
        args.ann_path = 'datasets/iu_xray/annotation.json'
        args.threshold = 3
        args.dataset_name = 'iu_xray'
        args.image_dir = 'datasets/iu_xray/images/'
        args.max_seq_length = 60
        args.d_model = 512
        args.d_ff = 512
        args.d_vf = 2048
        args.num_heads = 8
        args.num_layers = 3
        args.dropout = 0.1
        args.logit_layers = 1
        args.bos_idx = 0
        args.eos_idx = 0
        args.pad_idx = 0
        args.use_bn = True
        args.rm_num_slots = 3
        args.rm_num_heads = 8
        args.rm_d_model = 512
        args.visual_extractor = 'resnet101'
        args.visual_extractor_pretrained = True
        args.sample_method = 'beam_search'
        args.beam_size = 3
        args.temperature = 1.0
        args.sample_n = 1
        args.drop_prob_lm = 0.5
        
        # 创建tokenizer和模型
        tokenizer = Tokenizer(args)
        model = R2GenModel(args, tokenizer).cuda()
        print(f"✅ R2Gen模型创建成功")
        
        # 创建测试数据 (IU X-Ray格式: 2张图片)
        batch_size = 2
        images = torch.randn(batch_size, 2, 3, 224, 224).cuda()
        targets = torch.randint(1, 100, (batch_size, 20)).cuda()
        
        # 测试FP32
        print("🧪 测试FP32...")
        with torch.no_grad():
            output = model(images, targets, mode='train')
            print(f"✅ FP32成功: output shape={output.shape}")
        
        # 测试FP16
        print("🧪 测试FP16...")
        try:
            with debug_autocast(True, torch.float16):
                with torch.no_grad():
                    output = model(images, targets, mode='train')
                    print(f"✅ FP16成功: output shape={output.shape}")
        except Exception as e:
            print(f"❌ FP16失败: {e}")
            
            # 详细分析错误
            if "running_mean should contain" in str(e):
                print("🔍 检测到BatchNorm维度不匹配错误")
                print("这通常是由于预训练权重与当前模型结构不匹配导致的")
            elif "bias type" in str(e):
                print("🔍 检测到类型不匹配错误")
                print("这通常是由于autocast范围设置不当导致的")
            
            return False
            
    except Exception as e:
        print(f"❌ 完整模型测试失败: {e}")
        traceback.print_exc()
        return False
    
    return True

def analyze_batchnorm_layers():
    """分析模型中的BatchNorm层"""
    print("\n🔍 分析BatchNorm层...")
    
    try:
        from models.r2gen import R2GenModel
        from modules.tokenizers import Tokenizer
        import argparse
        
        # 创建测试参数
        args = argparse.Namespace()
        args.ann_path = 'datasets/iu_xray/annotation.json'
        args.threshold = 3
        args.dataset_name = 'iu_xray'
        args.image_dir = 'datasets/iu_xray/images/'
        args.max_seq_length = 60
        args.d_model = 512
        args.d_ff = 512
        args.d_vf = 2048
        args.num_heads = 8
        args.num_layers = 3
        args.dropout = 0.1
        args.logit_layers = 1
        args.bos_idx = 0
        args.eos_idx = 0
        args.pad_idx = 0
        args.use_bn = True
        args.rm_num_slots = 3
        args.rm_num_heads = 8
        args.rm_d_model = 512
        args.visual_extractor = 'resnet101'
        args.visual_extractor_pretrained = True
        args.sample_method = 'beam_search'
        args.beam_size = 3
        args.temperature = 1.0
        args.sample_n = 1
        args.drop_prob_lm = 0.5
        
        tokenizer = Tokenizer(args)
        model = R2GenModel(args, tokenizer)
        
        print("📊 BatchNorm层统计:")
        bn_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                print(f"  {name}: {type(module).__name__}, num_features={module.num_features}")
                print(f"    running_mean shape: {module.running_mean.shape}")
                print(f"    running_var shape: {module.running_var.shape}")
                bn_count += 1
        
        print(f"总计BatchNorm层数: {bn_count}")
        
    except Exception as e:
        print(f"❌ BatchNorm分析失败: {e}")
        traceback.print_exc()

def main():
    """主诊断函数"""
    print("🚀 开始BatchNorm兼容性问题诊断...")
    
    # 分析BatchNorm层
    analyze_batchnorm_layers()
    
    # 逐步测试各个组件
    ve_success = test_visual_extractor()
    ed_success = test_encoder_decoder()
    full_success = test_full_model()
    
    print(f"\n📋 诊断结果总结:")
    print(f"  Visual Extractor FP16: {'✅ 成功' if ve_success else '❌ 失败'}")
    print(f"  Encoder Decoder FP16: {'✅ 成功' if ed_success else '❌ 失败'}")
    print(f"  完整模型 FP16: {'✅ 成功' if full_success else '❌ 失败'}")
    
    if not any([ve_success, ed_success, full_success]):
        print("\n🔧 建议的修复策略:")
        print("1. 检查预训练权重的兼容性")
        print("2. 考虑使用LayerNorm替换BatchNorm")
        print("3. 调整autocast的作用范围")
        print("4. 使用更精确的类型转换")

if __name__ == "__main__":
    main()

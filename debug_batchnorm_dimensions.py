#!/usr/bin/env python3
"""
深入调试BatchNorm维度问题
"""

import sys
import torch
import torch.nn as nn
import traceback

sys.path.append('R2Gen-main')

def debug_att_embed_creation():
    """调试att_embed层的创建过程"""
    print("🔍 调试att_embed层创建...")
    
    try:
        from modules.tokenizers import Tokenizer
        from models.r2gen import R2GenModel
        import argparse
        
        # 创建完整的参数配置
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
        
        print(f"配置参数:")
        print(f"  d_vf (att_feat_size): {args.d_vf}")
        print(f"  d_model (input_encoding_size): {args.d_model}")
        print(f"  use_bn: {args.use_bn}")
        
        # 创建tokenizer
        tokenizer = Tokenizer(args)
        print(f"  vocab_size: {len(tokenizer.idx2token)}")
        
        # 创建模型并检查att_embed层
        model = R2GenModel(args, tokenizer)
        
        # 检查att_embed层的结构
        att_embed = model.encoder_decoder.att_embed
        print(f"\n📊 att_embed层结构:")
        for i, layer in enumerate(att_embed):
            print(f"  [{i}] {layer}")
            if isinstance(layer, nn.BatchNorm1d):
                print(f"      num_features: {layer.num_features}")
                print(f"      running_mean shape: {layer.running_mean.shape}")
                print(f"      running_var shape: {layer.running_var.shape}")
        
        # 测试数据流
        print(f"\n🧪 测试数据流...")
        batch_size = 2
        seq_len = 49
        
        # 创建测试输入
        att_feats = torch.randn(batch_size, seq_len, args.d_vf)
        print(f"输入att_feats形状: {att_feats.shape}")
        
        # 逐层测试
        x = att_feats
        for i, layer in enumerate(att_embed):
            print(f"\n  层 [{i}] {type(layer).__name__}:")
            print(f"    输入形状: {x.shape}")
            
            if isinstance(layer, nn.BatchNorm1d):
                # BatchNorm1d需要(N, C)或(N, C, L)格式
                print(f"    BatchNorm1d期望: (batch_size, num_features) 或 (batch_size, num_features, seq_len)")
                print(f"    当前输入: {x.shape}")
                
                # 检查是否需要reshape
                if len(x.shape) == 3:
                    # (batch_size, seq_len, features) -> (batch_size, features, seq_len)
                    x_reshaped = x.transpose(1, 2)
                    print(f"    转置后形状: {x_reshaped.shape}")
                    try:
                        output = layer(x_reshaped)
                        x = output.transpose(1, 2)  # 转回原格式
                        print(f"    输出形状: {x.shape}")
                    except Exception as e:
                        print(f"    ❌ BatchNorm失败: {e}")
                        break
                else:
                    try:
                        x = layer(x)
                        print(f"    输出形状: {x.shape}")
                    except Exception as e:
                        print(f"    ❌ BatchNorm失败: {e}")
                        break
            else:
                try:
                    x = layer(x)
                    print(f"    输出形状: {x.shape}")
                except Exception as e:
                    print(f"    ❌ 层失败: {e}")
                    break
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        traceback.print_exc()

def test_batchnorm_input_format():
    """测试BatchNorm1d的输入格式要求"""
    print("\n🔍 测试BatchNorm1d输入格式...")
    
    batch_size = 2
    seq_len = 49
    features = 2048
    
    # 创建BatchNorm1d
    bn = nn.BatchNorm1d(features)
    print(f"BatchNorm1d(num_features={features})")
    
    # 测试不同的输入格式
    formats = [
        ("(batch, features)", (batch_size, features)),
        ("(batch, features, seq)", (batch_size, features, seq_len)),
        ("(batch, seq, features)", (batch_size, seq_len, features)),
    ]
    
    for name, shape in formats:
        print(f"\n  测试格式 {name}: {shape}")
        x = torch.randn(shape)
        try:
            output = bn(x)
            print(f"    ✅ 成功: 输出形状 {output.shape}")
        except Exception as e:
            print(f"    ❌ 失败: {e}")

def check_model_state_dict():
    """检查模型状态字典中的BatchNorm参数"""
    print("\n🔍 检查模型状态字典...")
    
    try:
        from modules.tokenizers import Tokenizer
        from models.r2gen import R2GenModel
        import argparse
        
        # 创建参数
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
        
        # 检查att_embed相关的参数
        state_dict = model.state_dict()
        
        print("📊 att_embed相关参数:")
        for key, value in state_dict.items():
            if 'att_embed' in key:
                print(f"  {key}: {value.shape}")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 开始深入调试BatchNorm维度问题...")
    
    test_batchnorm_input_format()
    debug_att_embed_creation()
    check_model_state_dict()

if __name__ == "__main__":
    main()

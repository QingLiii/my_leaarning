import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    """
    R2Gen主模型类 - 基于Memory-driven Transformer的医学影像报告生成模型

    论文: "Generating Radiology Reports via Memory-driven Transformer"

    模型架构:
    1. Visual Extractor: 使用ResNet101提取图像特征
    2. Encoder-Decoder: 带有Relational Memory和MCLN的Transformer

    关键创新:
    - Relational Memory (RM): 记录生成过程中的模式信息
    - Memory-driven Conditional Layer Normalization (MCLN): 将memory融入decoder
    """
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        # 视觉特征提取器 - 使用ResNet101预训练模型
        self.visual_extractor = VisualExtractor(args)
        # 编码器-解码器 - 核心的Transformer架构，包含RM和MCLN
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        # 根据数据集选择不同的前向传播方法
        # IU X-Ray数据集每个病例有两张图片（正面+侧面）
        # MIMIC-CXR数据集每个病例通常只有一张图片
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        """返回模型信息，包括可训练参数数量"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        """
        IU X-Ray数据集的前向传播

        IU X-Ray特点: 每个病例包含两张图片（正面视图 + 侧面视图）
        需要将两张图片的特征进行拼接

        Args:
            images: [batch_size, 2, 3, H, W] - 两张X光图片
            targets: [batch_size, seq_len] - 目标报告序列（训练时使用）
            mode: 'train' 或 'sample' - 训练模式或推理模式

        Returns:
            output: 训练时返回logits，推理时返回生成的token序列
        """
        # 分别提取两张图片的特征
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])  # 第一张图片
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])  # 第二张图片

        # 拼接两张图片的特征
        # fc_feats: 全局特征，用于初始化
        # att_feats: 注意力特征，用于encoder-decoder attention
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if mode == 'train':
            # 训练模式：使用teacher forcing，输入目标序列
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            # 推理模式：自回归生成，使用beam search或greedy search
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        """
        MIMIC-CXR数据集的前向传播

        MIMIC-CXR特点: 每个病例通常只有一张图片

        Args:
            images: [batch_size, 3, H, W] - 单张X光图片
            targets: [batch_size, seq_len] - 目标报告序列（训练时使用）
            mode: 'train' 或 'sample' - 训练模式或推理模式

        Returns:
            output: 训练时返回logits，推理时返回生成的token序列
        """
        # 提取单张图片的特征
        att_feats, fc_feats = self.visual_extractor(images)

        if mode == 'train':
            # 训练模式：使用teacher forcing
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            # 推理模式：自回归生成
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output


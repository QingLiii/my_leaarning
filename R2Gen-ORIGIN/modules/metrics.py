from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge


def compute_scores(gts, res):
    """
    计算自然语言生成评估指标

    使用MS COCO评估工具包计算BLEU、METEOR、ROUGE-L分数

    Args:
        gts: 字典，键为图像ID，值为参考报告列表
             格式: {image_id: [ref_report1, ref_report2, ...]}
        res: 字典，键为图像ID，值为生成报告列表
             格式: {image_id: [generated_report]}

    Returns:
        eval_res: 包含各项评估指标的字典

    重要: BLEU分数异常可能的原因
    1. gts或res为空字典
    2. 生成的报告为空字符串
    3. 参考报告和生成报告格式不匹配
    4. tokenization问题导致无法匹配n-gram

    预期BLEU分数范围（根据论文）:
    - IU X-Ray: BLEU-4 ~0.165
    - MIMIC-CXR: BLEU-4 ~0.103
    """

    # 设置评估器
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),  # BLEU 1-4 gram
        (Meteor(), "METEOR"),                                   # METEOR分数
        (Rouge(), "ROUGE_L")                                    # ROUGE-L分数
    ]
    eval_res = {}

    # 为每个指标计算分数
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)

        if type(method) == list:
            # BLEU返回多个分数
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            # METEOR和ROUGE返回单个分数
            eval_res[method] = score
    return eval_res

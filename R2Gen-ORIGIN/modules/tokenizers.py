import json
import re
from collections import Counter


class Tokenizer(object):
    """
    医学报告分词器

    功能:
    1. 清理和预处理医学报告文本
    2. 构建词汇表
    3. 文本编码/解码

    关键特点:
    - 针对不同数据集使用不同的清理策略
    - 基于词频阈值构建词汇表
    - 处理医学报告的特殊格式（编号、标点等）

    潜在问题点:
    - 文本清理可能过于激进，丢失重要信息
    - 词汇表构建方式可能影响生成质量
    """
    def __init__(self, args):
        self.ann_path = args.ann_path          # 标注文件路径
        self.threshold = args.threshold        # 词频阈值，低于此值的词被标记为<unk>
        self.dataset_name = args.dataset_name  # 数据集名称

        # 根据数据集选择不同的文本清理方法
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr

        # 加载标注数据
        self.ann = json.loads(open(self.ann_path, 'r').read())
        # 构建词汇表
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        """
        将token ID序列解码为文本

        重要: 这个方法可能是BLEU分数异常的关键！

        解码逻辑:
        - 遇到ID=0时停止（0是padding或结束符）
        - 跳过ID<=0的token
        - 在token间添加空格

        潜在问题:
        - 如果生成的ID序列全是0或负数，会返回空字符串
        - 这可能导致BLEU分数为0或极小值
        """
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:  # 只处理有效的token ID
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break  # 遇到0或负数时停止
        return txt

    def decode_batch(self, ids_batch):
        """批量解码token ID序列"""
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

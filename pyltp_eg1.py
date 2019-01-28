# -*- coding: utf-8 -*-
"""
@refer: https://pyltp.readthedocs.io/zh_CN/latest/api.html#id21
@usage: pyltp 是 LTP 的 Python 封装，提供了分词，词性标注，命名实体识别，依存句法分析，语义角色标注的功能
"""
import os
from pyltp import SentenceSplitter, Segmentor, CustomizedSegmentor, Postagger, NamedEntityRecognizer,Parser, SementicRoleLabeller

class ltp_analyzer:
    def __init__(self, model_path):
        self.cws_model_path = os.path.join(model_path, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
        self.pos_model_path = os.path.join(model_path, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        self.ner_model_path = os.path.join(model_path, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        self.par_model_path = os.path.join(model_path, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

    def get_split_sentences(self, text):
        # 分句
        sents = SentenceSplitter.split(text)  # 分句
        print('\n'.join(sents))
        return list(sents)

    def get_cut_words(self, text, dict_path=None):
        # 分词
        segmentor = Segmentor()  # 初始化实例
        if dict_path is None:
            segmentor.load(self.cws_model_path)  # 加载模型
        else:
            segmentor.load_with_lexicon(self.cws_model_path, dict_path)  # 加载模型，第二个参数是您的外部词典文件路径
        words = segmentor.segment(text)
        print('\t'.join(words))
        segmentor.release()
        return list(words)

    def get_customized_cut_words(self, text, model_file):
        # 个性化分词模型
        customized_segmentor = CustomizedSegmentor()  # 初始化实例
        customized_segmentor.load(self.cws_model_path, model_file)  # 加载模型，第二个参数是您的增量模型路径
        words = customized_segmentor.segment(text)
        print('\t'.join(words))
        customized_segmentor.release()
        return list(words)

    def get_postags(self, words):
        postagger = Postagger()  # 初始化实例
        postagger.load(self.pos_model_path)  # 加载模型
        postags = postagger.postag(words)  # 词性标注
        print('\t'.join(postags))
        postagger.release()  # 释放模型
        return list(postags)

    def get_netags(self, words):
        # 命名实体识别
        postags = self.get_postags(words)
        recognizer = NamedEntityRecognizer()  # 初始化实例
        recognizer.load(self.ner_model_path)  # 加载模型
        netags = recognizer.recognize(list(words), list(postags))  # 命名实体识别
        print('\t'.join(netags))
        recognizer.release()  # 释放模型
        return list(netags)

    def get_dependency(self, words):
        # 句法分析
        postags = self.get_postags(words)
        parser = Parser()  # 初始化实例
        parser.load(self.par_model_path)  # 加载模型
        arcs = parser.parse(words, postags)  # 句法分析
        print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        parser.release()  # 释放模型
        return arcs

    def get_srl(self, words):
        # 语义角色标注
        labeller = SementicRoleLabeller()  # 初始化实例
        labeller.load(self.srl_model_path)  # 加载模型
        # arcs 使用依存句法分析的结果
        postags = self.get_postags(words)
        arcs = self.get_dependency(words)
        roles = labeller.label(words, postags, arcs)  # 语义角色标注

        # 打印结果
        for role in roles:
            print(role.index, "".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
        labeller.release()  # 释放模型
        return roles


def main():
    LTP_DATA_DIR = 'D:/Model/ltp_data_v3.4.0'  # ltp模型目录的路径
    text = '元芳你怎么看？我就趴窗口上看呗！'
    ltp0 = ltp_analyzer(model_path=LTP_DATA_DIR)
    ltp0.get_cut_words(text)




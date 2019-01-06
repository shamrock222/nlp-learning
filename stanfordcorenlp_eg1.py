# -*- coding: utf-8 -*-
"""
@refer: https://blog.csdn.net/guolindonggld/article/details/72795022
@usage: 使用Stanford CoreNLP的Python封装包处理中文
"""

from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'./model/stanford-corenlp-full-2016-10-31/', lang='zh')

sentence = '清华大学位于北京。'
print(nlp.word_tokenize(sentence))
print(nlp.pos_tag(sentence))
print(nlp.ner(sentence))
print(nlp.parse(sentence))
print(nlp.dependency_parse(sentence))

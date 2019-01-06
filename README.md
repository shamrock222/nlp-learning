# nlp-learning
## 0 tools
### standford corenlp
使用Stanford CoreNLP的Python封装包处理中文 https://blog.csdn.net/guolindonggld/article/details/72795022  
安装stanfordcorenlp包:   
  1)下载安装JDK 1.8及以上版本  
  2)下载Stanford CoreNLP文件，解压  
  3)处理中文还需要下载中文模型的jar文件，然后放到CoreNLP解压包根目录下面   
  4) pip install stanfordcorenlp   
## 1 spell correction:  
How to Write a Spelling Corrector   http://norvig.com/spell-correct.html  
中文(语音结果)的文本纠错综述 Chinese Spelling Check  https://blog.csdn.net/lipengcn/article/details/82556569  
### 1.1 symspell
SymSpell  https://github.com/wolfgarbe/SymSpell  
SymSpellpy  https://github.com/mammothb/symspellpy    
pip3 install -U symspellpy 
### 1.2 pycorrector 
中文文本纠错工具。音似、形似错字（或变体字）纠正，可用于中文拼音、笔画输入法的错误纠正。python3开发。 
pycorrector依据语言模型检测错别字位置，通过拼音音似特征、笔画五笔编辑距离特征及语言模型困惑度特征纠正错别字。  
pycorrector: https://github.com/shibing624/pycorrector  
pip3 install -r requirements.txt  
pip3 install pycorrector  
### 1.3 pypinyin
## 2 句法分析  
CS224n之句法分析总结 https://blog.csdn.net/yu5064/article/details/82151578  
### 2.1 依存关系分析(dependency parsing)
基于神经网络的依存句法分析总结  https://blog.csdn.net/yu5064/article/details/82186738




# -*- coding: utf-8 -*-

import json
import jieba
from textqa.model.CQA import args
from textqa.model.CQA.utils import Utils
from textqa.model.CQA.file_pool import FilePool


# 数据预处理
class PreProcessor:
    def __init__(self):
        # 停用词表
        with open(FilePool.stopword_txt, 'r') as f_stopword:
            doc = f_stopword.readlines()
        self.stopwords = [line.rstrip('\n') for line in doc]

        # 关键词库
        with open(FilePool.keyword_database_json, 'r') as f_kwdb:
            self.keyword_database = json.load(f_kwdb)

    # 清洗text
    def clean_input(self, text):
        """
        Returns
        -------
        cleaned : list of list of str
            清洗、分词后的输入
        uncut : list of str
            仅清洗后的输入(没有分词)
        """
        cleaned = []
        uncut = []
        for line in text:
            line = Utils.str2cn(line)  # 阿拉伯数字转中文
            cut_line = [w for w in jieba.cut(line)]  # 对query进行分词
            # 去停用词
            if args.trim_stop:
                trim_line = [w for w in cut_line if w not in self.stopwords]
            else:
                trim_line = cut_line

            uncut_line = ''.join(trim_line)
            cleaned.append(trim_line)
            uncut.append(uncut_line)
        return cleaned, uncut

    # 对问题进行分类
    def categorize(self, questions):
        categories = []  # 问题所属的类别，可能有多个
        answers = []
        for ques in questions:
            for dict in self.keyword_database:
                for word in dict['keywords']:
                    # 如果有关键词在问题里出现了，那么说明问题属于这个类别
                    if word in ques:
                        categories.append(dict['class'])
                        answers += dict['answers']
                        break  # 没必要再在同样的类别下面纠结了

        return categories, answers

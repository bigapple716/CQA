# -*- coding: utf-8 -*-


# 已分类的问题&答案
class CategorizedQA:
    def __init__(self, question):
        self.question = question
        self.categories = []  # 问题所属的类别，可能有多个
        self.uncut_answers = []  # 未分词的分类答案
        self.cut_answers = []  # 已分词的分类答案

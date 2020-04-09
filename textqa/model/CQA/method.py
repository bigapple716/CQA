# -*- coding: utf-8 -*-

from enum import Enum


class Method(Enum):
    bm25 = 0  # 普通的bm25
    bm25_syn = 1  # bm25 + 近义词替换
    bm25_new = 2  # 改进版的bm25
    qq_match = 3  # 问题-问题匹配
    mix = 4  # QQ匹配 + QA匹配
    tfidf_sim = 5  # TF-IDF向量相似度

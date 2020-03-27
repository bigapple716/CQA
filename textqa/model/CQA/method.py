# -*- coding: utf-8 -*-

from enum import Enum


class Method(Enum):
    bm25 = 0
    bm25_syn = 1
    bm25_new = 2
    qq_match = 3
    mix = 4
    tfidf_sim = 5
    aver_embed = 6
    lm = 7

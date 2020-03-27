# -*- coding: utf-8 -*-

import math
from six import iteritems
from six.moves import range
from functools import partial
from multiprocessing import Pool
from gensim.utils import effective_n_jobs
from gensim.summarization.bm25 import BM25
import synonyms

PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class NewBM25(BM25):
    # 继承父类的构造方法
    def __init__(self, corpus):
        super(NewBM25, self).__init__(corpus)

    def get_new_score(self, document, expanded_document, index):
        """
        Computes BM25 score of given `document` in relation to item of corpus selected by `index`.

        Parameters
        ----------
        document : list of str
            Document to be scored.
        index : int
            Index of document in corpus selected to score with `document`.

        Returns
        -------
        float
            BM25 score.

        """
        score = 0
        doc_freqs = self.doc_freqs[index]
        for word in expanded_document:
            if word not in doc_freqs:
                continue
            score += doc_freqs[word] * self.idf[word] * self.__delta(word, document)

        return score

    def get_new_scores(self, document, expanded_document):
        scores = [self.get_new_score(document, expanded_document, index) for index in range(self.corpus_size)]
        return scores

    def __delta(self, i, q):
        if i in q:
            return 1
        else:
            return self.__max_sim(i, q)

    def __max_sim(self, i, document):
        sim_list = [synonyms.compare(i, d, seg=False) for d in document]
        return max(sim_list)

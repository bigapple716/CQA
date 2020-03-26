# -*- coding: utf-8 -*-

import logging
from functools import partial
import re
from gensim import interfaces, matutils, utils
from gensim.utils import deprecated
from six import iteritems, iterkeys
import numpy as np
from gensim.models import TfidfModel


class NewTfidf(TfidfModel):
    def __init__(self, corpus=None):
        super(NewTfidf, self).__init__(corpus)

    def __getitem__(self, bow, eps=1e-12):
        self.eps = eps
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        # unknown (new) terms will be given zero weight (NOT infinity/huge weight,
        # as strict application of the IDF formula would dictate)

        termid_array, tf_array = [], []
        for termid, tf in bow:
            termid_array.append(termid)
            tf_array.append(tf)

        tf_array = self.wlocal(np.array(tf_array))

        vector = [
            (termid, tf * self.idfs.get(termid))
            for termid, tf in zip(termid_array, tf_array) if abs(self.idfs.get(termid, 0.0)) > self.eps
        ]

        # and finally, normalize the vector either to unit length, or use a
        # user-defined normalization function
        self.normalize = matutils.unitvec

        norm_vector = self.normalize(vector)

        norm_vector = [(termid, weight) for termid, weight in norm_vector if abs(weight) > self.eps]

        return norm_vector

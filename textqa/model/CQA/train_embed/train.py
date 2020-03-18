# -*- coding: utf-8 -*-

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
import json

# constant
model_file = 'word2vec.model'
word2vec_file = 'word2vec.kv'
corpus_file = '../data/cleaned_answers.json'

if __name__ == '__main__':
    with open(corpus_file, 'r') as f_corpus:
        corpus = json.load(f_corpus)
    model = Word2Vec(corpus, size=100, min_count=1, workers=12)
    model.save(model_file)  # save model
    model.wv.save(word2vec_file)  # save vectors
    # test
    wv = KeyedVectors.load(word2vec_file, mmap='r')
    vector = wv['行李']  # numpy vector of a word
    print(vector)

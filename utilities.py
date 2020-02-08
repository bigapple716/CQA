# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors
import pickle


# 统计文件里平均行长度
def aver_len(input_file):
    with open(input_file, 'r') as f_in:
        text = f_in.readlines()

    total_len = 0
    for line in text:
        total_len += len(line.rstrip())
    return total_len / len(text)


# load word2vec text file and dump it to pickle
def load_embed(word2vec_file):
    word2vec = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    with open('data/word2vec.pickle', 'wb') as f_pickle:
        pickle.dump(word2vec, f_pickle)


print(aver_len('data/input.txt'))
print(aver_len('data/cleaned_answers.txt'))
print(aver_len('data/long_answers.txt'))

load_embed('data/merge_sgns_bigram_char300.txt')

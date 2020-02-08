# -*- coding: utf-8 -*-

import json
from gensim.summarization.bm25 import *
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.similarities import MatrixSimilarity
from gensim.matutils import jaccard
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import torch
import pickle

from models import *
from pytorch_pretrained import BertTokenizer
from utils import Utils

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


class Baselines:
    def __init__(self, ans_json, word2vec_file):
        with open(ans_json, 'r') as f_json:
            self.cut_answers = f_json.readlines()
        with open(word2vec_file, 'r') as f_pickle:
            self.word2vec = pickle.load(f_pickle)


    # bm25算法搜索
    @staticmethod
    def bm25(query, answers_json):
        corpus = [query]
        # read from json file
        with open(answers_json, mode='r', encoding='utf-8') as f_ans_json:
            corpus += json.load(f_ans_json)
        bm25_weights = get_bm25_weights(corpus)[0]
        bm25_weights.pop(0)  # 去掉第一个元素(即query)

        sorted_scores = sorted(bm25_weights, reverse=True)  # 将得分从大到小排序
        max_pos = np.argsort(bm25_weights)[::-1]  # 从大到小排序，返回index(而不是真正的value)
        # max_pos = Utils.trim_result(sorted_scores, max_pos, threshold=10)
        return max_pos

    # tf-idf相似度算法搜索
    @staticmethod
    def tfidf_sim(query, answers_json):
        # read from json file
        with open(answers_json, mode='r', encoding='utf-8') as f_ans:
            corpus = json.load(f_ans)
        # 构造bag of words
        dict = Dictionary(corpus)  # fit dictionary
        bow = [dict.doc2bow(line) for line in corpus]  # convert corpus to BoW format
        query_bow = [dict.doc2bow(query)]
        # 构造tf-idf模型
        model = TfidfModel(bow)  # fit model
        tfidf_weights = model[bow]  # apply model
        query_tfidf = model[query_bow]
        sim_index = MatrixSimilarity(tfidf_weights)
        max_pos = np.argsort(sim_index[query_tfidf][0])[::-1]  # 从大到小排序，返回index(而不是真正的value)
        return max_pos

    # tf-idf算法搜索
    @staticmethod
    def tfidf_dist(query, answers_json):
        # read from json file
        with open(answers_json, mode='r', encoding='utf-8') as f_ans:
            corpus = json.load(f_ans)
        # 构造bag of words
        dict = Dictionary(corpus)  # fit dictionary
        bow = [dict.doc2bow(line) for line in corpus]  # convert corpus to BoW format
        query_bow = [dict.doc2bow(query)]
        # 构造tf-idf模型
        model = TfidfModel(bow)  # fit model
        tfidf_weights = model[bow]  # apply model
        query_tfidf = model[query_bow]
        distances = [jaccard(query_tfidf[0], tfidf_weights[i]) for i in range(len(tfidf_weights))]  # 计算Jaccard距离
        max_pos = np.argsort(distances)  # 从小到大排序(注意不是从大到小，因为Jaccard距离越小越近)
        return max_pos

    # tf-idf
    @staticmethod
    def tfidf(query, answers_txt):
        cut_query = ' '.join(query)  # 将列表形式的query转化成空格隔开的形式
        # read from txt file
        with open(answers_txt, mode='r', encoding='utf-8') as f_ans:
            corpus = f_ans.readlines()
        # 将列表形式的corpus转化成空格隔开的形式
        cut_corpus = []
        for line in corpus:
            cut_corpus.append(' '.join(jieba.cut(line.rstrip())))

        doc_score = [0] * len(corpus)  # 每个doc的得分 = query里每个词在这个doc里面的得分之和

        model = TfidfVectorizer(token_pattern=r'\b\w+\b')  # 实例化一个TfidfVectorizer
        model.fit(cut_corpus)
        query_tokens = model.transform([cut_query]).indices  # 将query转化为token表示
        for doc_id in range(len(cut_corpus)):  # 遍历每个doc
            score = 0  # 初始得分为0
            doc_tfidf = model.transform([cut_corpus[doc_id]])  # 计算doc里每个词在这个doc里的tfidf得分
            doc_tokens = doc_tfidf.indices  # 将doc转化为token表示
            for token in query_tokens:  # 遍历query里的每个词
                # 如果query里的这个词在这个doc里面，那么tfidf得分就不是0，可以加到score里
                if token in doc_tokens:
                    score += doc_tfidf[0, token]
            doc_score[doc_id] = score

        sorted_scores = sorted(doc_score, reverse=True)  # 将得分从大到小排序
        max_pos = np.argsort(doc_score)[::-1]  # 从大到小排序，返回index(而不是真正的value)
        # max_pos = Utils.trim_result(sorted_scores, max_pos, threshold=0.5)
        return max_pos

    # 词向量平均
    def aver_embed(self, query):
        doc_score = []

        words = [w for w in query if w in self.word2vec.vocab]  # remove out-of-vocabulary words
        query_token = np.mean(self.word2vec[words], axis=0)  # average embedding of words in the query
        for ans in self.cut_answers:
            words = [w for w in ans if w in self.word2vec.vocab]
            ans_token = np.mean(self.word2vec[words], axis=0)
            doc_score.append(cosine_similarity(query_token, ans_token))

        sorted_scores = sorted(doc_score, reverse=True)  # 将得分从大到小排序
        max_pos = np.argsort(doc_score)[::-1]  # 从大到小排序，返回index(而不是真正的value)
        return max_pos


class NeuralNetworks:
    def __init__(self):
        # set device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Pytorch is using', device)

        self.bert_tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')
        self.ernie_tokenizer = BertTokenizer.from_pretrained('./ERNIE_pretrain')

        self.bert_model = Bert()
        self.ernie_model = Ernie()

        self.bert_model.eval()
        self.ernie_model.eval()

    def bert_search(self, query, answers_txt):
        scores = []  # 每个答案的得分

        # 将query转化成tensor的形式
        token_q = [CLS] + self.bert_tokenizer.tokenize(query)
        token_id_q = self.bert_tokenizer.convert_tokens_to_ids(token_q)
        tensor_q = torch.tensor(token_id_q).unsqueeze(0)

        # read from txt file
        with open(answers_txt, mode='r', encoding='utf-8') as f_ans:
            corpus = f_ans.readlines()

        # 遍历答案库
        for a in corpus:
            a = a.rstrip()
            # 将答案a转化成tensor的形式
            token_a = [CLS] + self.bert_tokenizer.tokenize(a)
            token_id_a = self.bert_tokenizer.convert_tokens_to_ids(token_a)
            tensor_a = torch.tensor(token_id_a).unsqueeze(0)

            # 计算query和a的相似度
            score_a = self.bert_model(tensor_q, tensor_a).item()
            scores.append(score_a)

        max_pos = np.argsort(scores)[::-1]  # 从大到小排序，返回index(而不是真正的value)
        return max_pos

    def ernie_search(self, query, answers_txt):
        scores = []  # 每个答案的得分

        # 将query转化成tensor的形式
        token_q = [CLS] + self.ernie_tokenizer.tokenize(query)
        token_id_q = self.ernie_tokenizer.convert_tokens_to_ids(token_q)
        tensor_q = torch.tensor(token_id_q).unsqueeze(0)

        # read from txt file
        with open(answers_txt, mode='r', encoding='utf-8') as f_ans:
            corpus = f_ans.readlines()

        # 遍历答案库
        for a in corpus:
            a = a.rstrip()
            # 将答案a转化成tensor的形式
            token_a = [CLS] + self.ernie_tokenizer.tokenize(a)
            token_id_a = self.ernie_tokenizer.convert_tokens_to_ids(token_a)
            tensor_a = torch.tensor(token_id_a).unsqueeze(0)

            # 计算query和a的相似度
            score_a = self.ernie_model(tensor_q, tensor_a).item()
            scores.append(score_a)

        max_pos = np.argsort(scores)[::-1]  # 从大到小排序，返回index(而不是真正的value)
        return max_pos

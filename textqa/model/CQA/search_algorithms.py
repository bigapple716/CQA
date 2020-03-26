# -*- coding: utf-8 -*-

import json
from gensim.summarization.bm25 import BM25
from gensim.models import TfidfModel, KeyedVectors
from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import synonyms
from nltk.lm.preprocessing import *
from nltk.lm.models import KneserNeyInterpolated
from textqa.model.CQA.file_pool import FilePool
from textqa.model.CQA import args
from textqa.model.CQA.new_tfidf import NewTfidf

self_trained_word2vec = 'train_embed/word2vec.kv'


class Baselines:
    def __init__(self, use_aver_embed=False, use_pretrained_word2vec=True):
        # 读入停用词表
        with open(FilePool.stopword_txt, 'r') as f_stopword:
            doc = f_stopword.readlines()
        self.stopwords = [line.rstrip('\n') for line in doc]

        # 读入答案
        if args.answer_base == 'long':
            # 使用长答案
            ans_json = FilePool.long_answers_json
            ans_txt = FilePool.long_answers_txt
        elif args.answer_base == 'cleaned':
            # 使用短答案
            ans_json = FilePool.cleaned_answers_json
            ans_txt = FilePool.cleaned_answers_txt
        else:
            # 使用small answers
            ans_json = FilePool.small_answers_json
            ans_txt = FilePool.small_answers_txt
        with open(ans_json, 'r') as f_json:
            text = json.load(f_json)
            if args.trim_stop:
                self.cut_answers = [[ele for ele in answer if ele not in self.stopwords] for answer in text]
            else:
                self.cut_answers = text
        with open(ans_txt, 'r') as f_ans_txt:
            text = f_ans_txt.readlines()
            self.uncut_answers = [line.rstrip('\n') for line in text]

        with open(FilePool.qa_file, 'r') as f_qa:
            self.base_questions = json.load(f_qa)

        with open(FilePool.base_ques_list_file, 'r') as f_base_ques_list:
            self.base_ques_list = json.load(f_base_ques_list)

        # 提前实例化bm25模型，提升性能
        # 如果提前对问题分类了，那么没必要提前实例化模型，因为每个问题对应的答案库都不一样
        if not args.categorize_question:
            self.bm25_model = BM25(self.cut_answers)

        # 提前实例化tfidf模型，提升性能
        if args.method == 'mix' or 'qq-match':
            self.tfidf_dict = Dictionary(self.base_ques_list)  # fit dictionary
            n_features = len(self.tfidf_dict.token2id)
            bow = [self.tfidf_dict.doc2bow(line) for line in self.base_ques_list]  # convert corpus to BoW format
            # 构造tf-idf模型
            self.tfidf_model = TfidfModel(bow)  # fit model
            text_tfidf = self.tfidf_model[bow]  # apply model
            self.sim_index = SparseMatrixSimilarity(text_tfidf, n_features)
        else:
            self.tfidf_dict = Dictionary(self.cut_answers)  # fit dictionary
            n_features = len(self.tfidf_dict.token2id)
            bow = [self.tfidf_dict.doc2bow(line) for line in self.cut_answers]  # convert corpus to BoW format
            # 构造tf-idf模型
            self.tfidf_model = TfidfModel(bow)  # fit model
            text_tfidf = self.tfidf_model[bow]  # apply model
            self.sim_index = SparseMatrixSimilarity(text_tfidf, n_features)

        if use_aver_embed:
            if use_pretrained_word2vec:
                # 用预训练好的word2vec
                with open(FilePool.word2vec_pickle, 'rb') as f_pickle:
                    self.word2vec = pickle.load(f_pickle)
            else:
                # 用机场文档训练出的word2vec
                self.word2vec = KeyedVectors.load(self_trained_word2vec, mmap='r')

    # bm25算法搜索
    def bm25(self, query, categorized_qa):
        # 只有问题分类的情况下才在这里做模型实例化，其他情况下模型已经在__init__()里实例化过了
        if args.categorize_question:
            if len(categorized_qa['cut_answers']) != 0:
                # 非空的时候才用这个作corpus传进BM25
                self.bm25_model = BM25(categorized_qa['cut_answers'])
                # print(categorized_qa['classes'])
            else:
                # 如果为空，那么还用原来的corpus传进BM25
                self.bm25_model = BM25(self.cut_answers)
                # print('没用分类问题')

        bm25_weights = self.bm25_model.get_scores(query)

        sorted_scores = sorted(bm25_weights, reverse=True)  # 将得分从大到小排序
        sorted_scores = [s / (len(query) + 1) for s in sorted_scores]  # 将得分除以句长
        max_pos = np.argsort(bm25_weights)[::-1]  # 从大到小排序，返回index(而不是真正的value)

        # 根据max_pos从答案库里把真正的答案抽出来
        if args.categorize_question:
            # 答案来源是categorized的时候
            if len(categorized_qa['cut_answers']) != 0:
                # 非空的时候才用这个作为answer base
                answers = self.__max_pos2answers(max_pos, categorized_qa['uncut_answers'])
            else:
                # 如果为空，那么还用原来的self.uncut_answers作为answer base
                answers = self.__max_pos2answers(max_pos, self.uncut_answers)
        else:
            # 答案来源不是categorized的时候，categorized_qa是None
            answers = self.__max_pos2answers(max_pos, self.uncut_answers)

        return sorted_scores, max_pos, answers

    # bm25 with synonym module
    def bm25_syn(self, query):
        query_weights = self.bm25_model.get_scores(query)  # 普通的bm25算法
        max_pos = np.argsort(query_weights)[::-1][0]  # 最高得分所在的index(而不是真正的value)

        # 找出来query里哪个词是最关键的
        max_score = 0
        kw = ''  # 最关键的那个词
        kw_idx = -1
        for idx, word in enumerate(query):
            word_weight = self.bm25_model.get_score([word], index=max_pos)
            if word_weight > max_score:
                max_score = word_weight
                kw = word
                kw_idx = idx

        # 为这个最关键的词创造一个近义词列表
        nearby_list = synonyms.nearby(kw)
        syn_list = [kw]  # 先手动把关键词自己加到列表里
        for word, score in zip(nearby_list[0], nearby_list[1]):
            # 条件：得分大于阈值
            if score > args.syn_threshold:
                syn_list.append(word)

        # 找出来哪个近义词得分最高
        max_score = -1
        best_kw = ''  # 得分最高的词
        for syn in syn_list:
            query[kw_idx] = syn  # 替换query中的那个最关键的词
            weights = self.bm25_model.get_scores(query)  # 普通的bm25算法
            score = sorted(weights, reverse=True)[0]  # 将得分从大到小排序，取第1个
            if score > max_score:
                max_score = score
                best_kw = syn

        # if best_kw != kw:
        #     print('1')
        # else:
        #     print('0')
        # print(kw + '\t' + best_kw)

        # 找到最合适的关键词了，回到正规，返回sorted_scores, max_pos, answers
        query[kw_idx] = best_kw
        bm25_weights = self.bm25_model.get_scores(query)

        sorted_scores = sorted(bm25_weights, reverse=True)  # 将得分从大到小排序
        sorted_scores = [s / (len(query) + 1) for s in sorted_scores]  # 将得分除以句长
        max_pos = np.argsort(bm25_weights)[::-1]  # 从大到小排序，返回index(而不是真正的value)
        answers = self.__max_pos2answers(max_pos, self.uncut_answers)

        return sorted_scores, max_pos, answers

    # 问题-问题匹配
    def qq_match(self, query):
        # 输入tf-idf，得到从大到小排列的index list
        sorted_scores, max_pos, _ = self.tfidf_sim(query)
        answers, questions = self.__max_pos2answers_questions(max_pos)

        # 用QQ匹配的阈值过滤一遍结果
        sorted_scores, max_pos, answers, questions = \
            self.__filter_by_threshold(sorted_scores, max_pos, answers, questions, args.qq_threshold)

        return sorted_scores, max_pos, answers, questions

    # QQ匹配和QA匹配混合
    def qq_qa_mix(self, query, categorized_qa):
        sorted_scores, max_pos, answers, questions = self.qq_match(query)  # 先用QQ匹配试试

        if len(sorted_scores) > 0:
            # QQ匹配效果不错，直接返回结果
            return sorted_scores, max_pos, answers, questions
        else:
            # 截断之后啥也不剩了，说明QQ匹配没有一个得分到阈值的
            # 果断放弃，改用QA匹配
            # QA匹配暂时选用bm25算法
            sorted_scores, max_pos, answers = self.bm25(query, categorized_qa)

            # 用QA匹配的阈值过滤一遍结果
            sorted_scores, max_pos, answers, _ = \
                self.__filter_by_threshold(sorted_scores, max_pos, answers, [], args.qa_threshold)

            return sorted_scores, max_pos, answers, []  # questions的位置返回一个空list

    # tf-idf相似度算法搜索
    def tfidf_sim(self, query):
        query_bow = [self.tfidf_dict.doc2bow(query)]  # 用query做一个bag of words
        query_tfidf = self.tfidf_model[query_bow]  # 用tfidf model编码
        similarities = self.sim_index[query_tfidf][0]  # 算相似度

        sorted_scores = sorted(similarities, reverse=True)  # 将得分从大到小排序
        max_pos = np.argsort(similarities)[::-1]  # 从大到小排序，返回index(而不是真正的value)
        answers = self.__max_pos2answers(max_pos, self.uncut_answers)  # 根据max_pos从答案库里把真正的答案抽出来
        return sorted_scores, max_pos, answers

    def new_tfidf(self, query):
        # 实例化new-tfidf模型
        self.tfidf_dict = Dictionary(self.cut_answers)  # fit dictionary
        n_features = len(self.tfidf_dict.token2id)
        bow = [self.tfidf_dict.doc2bow(line) for line in self.cut_answers]  # convert corpus to BoW format
        # 构造tf-idf模型
        self.tfidf_model = NewTfidf(bow)  # fit model
        text_tfidf = self.tfidf_model[bow]  # apply model
        self.sim_index = SparseMatrixSimilarity(text_tfidf, n_features)

        query_bow = [self.tfidf_dict.doc2bow(query)]  # 用query做一个bag of words
        query_tfidf = self.tfidf_model[query_bow]  # 用tfidf model编码
        similarities = self.sim_index[query_tfidf][0]  # 算相似度

        sorted_scores = sorted(similarities, reverse=True)  # 将得分从大到小排序
        max_pos = np.argsort(similarities)[::-1]  # 从大到小排序，返回index(而不是真正的value)
        answers = self.__max_pos2answers(max_pos, self.uncut_answers)  # 根据max_pos从答案库里把真正的答案抽出来
        return sorted_scores, max_pos, answers

    # 词向量平均(暂停维护)
    def aver_embed(self, query):
        doc_score = []

        words = [w for w in query if w in self.word2vec.vocab]  # remove out-of-vocabulary words
        query_token = np.mean(self.word2vec[words], axis=0)  # average embedding of words in the query
        for ans in self.cut_answers:
            words = [w for w in ans if w in self.word2vec.vocab]
            ans_token = np.mean(self.word2vec[words], axis=0)
            cos_sim = cosine_similarity(query_token.reshape(1, -1), ans_token.reshape(1, -1))
            doc_score.append(np.asscalar(cos_sim))

        sorted_scores = sorted(doc_score, reverse=True)  # 将得分从大到小排序
        max_pos = np.argsort(doc_score)[::-1]  # 从大到小排序，返回index(而不是真正的value)
        answers = self.__max_pos2answers(max_pos, self.uncut_answers)  # 根据max_pos从答案库里把真正的答案抽出来
        return sorted_scores, max_pos, answers

    # Language Model(暂停维护)
    def language_model(self, query):
        doc_score = []
        for text in self.cut_answers:
            train, vocab = padded_everygram_pipeline(order=1, text=text)

            lm = KneserNeyInterpolated(1)  # 实例化模型
            lm.fit(train, vocab)  # 喂训练数据

            score = 1
            for word in query:
                score *= lm.score(word)

            doc_score.append(score)

        sorted_scores = sorted(doc_score, reverse=True)  # 将得分从大到小排序
        max_pos = np.argsort(doc_score)[::-1]  # 从大到小排序，返回index(而不是真正的value)
        answers = self.__max_pos2answers(max_pos, self.uncut_answers)  # 根据max_pos从答案库里把真正的答案抽出来
        return sorted_scores, max_pos, answers

    # 根据max_pos从答案库里把真正的答案抽出来
    def __max_pos2answers(self, max_pos, uncut_answers):
        max_pos = max_pos.tolist()  # ndarray -> list
        answers = []
        for r in max_pos:
            answers.append(uncut_answers[r])
        return answers

    def __max_pos2answers_questions(self, max_pos):
        max_pos = max_pos.tolist()  # ndarray -> list
        answers = []
        questions = []
        for r in max_pos:
            answers.append(self.base_questions[r]['sentence'])
            questions.append(self.base_questions[r]['question'])
        return answers, questions

    # 根据阈值对结果进行截断
    def __filter_by_threshold(self, sorted_scores, max_pos, answers, questions, threshold):
        cut_point = 10000  # 截断位置，先初始化一个很大的值
        for i, score in enumerate(sorted_scores):
            # 如果第i个score小于阈值，那么后面的一定都小于阈值，从这里截断就行
            if score < threshold:
                cut_point = i  # 把截断位置设为i
                break
        # 进行截断操作
        sorted_scores = sorted_scores[:cut_point]
        max_pos = max_pos[:cut_point]
        answers = answers[:cut_point]
        questions = questions[:cut_point]
        return sorted_scores, max_pos, answers, questions

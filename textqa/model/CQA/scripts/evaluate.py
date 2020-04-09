# -*- coding: utf-8 -*-

import json
import csv
import numpy as np
from gensim.models import TfidfModel, KeyedVectors
from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity
from textqa.model.CQA.file_pool import FilePool
from textqa.model.CQA.utils import Utils
from textqa.model.CQA.pre_processor import PreProcessor


# 自动化测试
class Evaluate:
    def __init__(self):
        self.eval_res_file = 'textqa/model/CQA/scripts/eval_result.txt'

        self.pre_processor = PreProcessor()

        # 读入停用词表
        with open(FilePool.stopword_txt, 'r') as f_stopword:
            doc = f_stopword.readlines()
        self.stopwords = [line.rstrip('\n') for line in doc]

        # 读入问题
        with open(FilePool.input_txt, 'r') as f_q:
            doc = f_q.readlines()
            self.queries = Utils.clean_text(doc)  # 清洗输入

        # 读入top3的回答
        with open(FilePool.output_csv, 'r') as f_res:
            reader = csv.reader(f_res, lineterminator='\n')
            self.top3_answers = []
            self.valid_answer_size = 0
            for row in reader:
                if row[0] != '-1':
                    self.valid_answer_size += 1
                row = Utils.clean_text(row)
                self.top3_answers.append(row)

        # 读入问答对
        with open(FilePool.whole_qa_file, 'r') as f_qa:
            self.qa = json.load(f_qa)
            # 清洗问题和答案
            for dict in self.qa:
                dict['question'] = Utils.clean_line(dict['question'])
                dict['sentence'] = Utils.clean_text(dict['sentence'])

        # 校验问题数量和答案数量是否相等
        if len(self.queries) != len(self.top3_answers):
            raise Exception('问题和答案数量不相等！')

        # 读入答案库
        with open(FilePool.small_answers_json, 'r') as f_json:
            text = json.load(f_json)
            self.cut_answers = [[ele for ele in answer if ele not in self.stopwords] for answer in text]

        # 提前实例化tfidf-sim模型
        self.tfidf_dict = Dictionary(self.cut_answers)  # fit dictionary
        self.n_features = len(self.tfidf_dict.token2id)
        bow = [self.tfidf_dict.doc2bow(line) for line in self.cut_answers]  # convert corpus to BoW format
        self.tfidf_model = TfidfModel(bow)  # fit model

    def evaluate(self):
        hit1 = 0
        hit3 = 0
        found = 0
        hit1_result = []
        hit3_result = []
        gold_answers = []

        for query, ans_list in zip(self.queries, self.top3_answers):
            has_found = False
            has_hit1 = False
            has_hit3 = False
            first_gold = ''
            for dict in self.qa:  # 遍历QA库
                if query == dict['question']:
                    # 在QA库里找到这个问题了，看看有没有gold在top1/top3里面的
                    found += 1
                    has_found = True
                    first_gold = dict['sentence'][0]
                    for gold in dict['sentence']:
                        if self.__match(gold, [ans_list[0]]):
                            hit1 += 1
                            has_hit1 = True
                        if self.__match(gold, ans_list):
                            hit3 += 1
                            has_hit3 = True
                            break
                    break
            if not has_found:
                print('query not found:', query)

            if has_hit1:
                hit1_result.append('1')
            else:
                hit1_result.append('0')
            if has_hit3:
                hit3_result.append('1')
            else:
                hit3_result.append('0')
            gold_answers.append(first_gold)

        # 输出结果
        print('hit1:', hit1 / self.valid_answer_size)
        print('hit3:', hit3 / self.valid_answer_size)
        print('queries found in QA base:', found)
        with open(self.eval_res_file, 'w') as f_res:
            f_res.write('hit1' + '\t' + 'hit3' + '\t' + 'gold_answer' + '\n')
            for h1, h3, g in zip(hit1_result, hit3_result, gold_answers):
                f_res.write(h1 + '\t' + h3 + '\t' + g + '\n')

    # 从txt文件中获取gold，进行评价
    def evaluate_from_txt(self):
        gold_file1 = 'textqa/model/CQA/scripts/tmp_gold1.txt'
        gold_file2 = 'textqa/model/CQA/scripts/tmp_gold2.txt'
        gold_file3 = 'textqa/model/CQA/scripts/tmp_gold3.txt'
        query_file = 'textqa/model/CQA/data/input.txt'
        result_file = 'textqa/model/CQA/scripts/results.txt'

        with open(gold_file1, 'r') as f_gold1:
            doc = f_gold1.readlines()
            gold1 = [line.rstrip('\n') for line in doc]
        with open(gold_file2, 'r') as f_gold2:
            doc = f_gold2.readlines()
            gold2 = [line.rstrip('\n') for line in doc]
        with open(gold_file3, 'r') as f_gold3:
            doc = f_gold3.readlines()
            gold3 = [line.rstrip('\n') for line in doc]
        with open(query_file, 'r') as f_q:
            doc = f_q.readlines()
            queries = [line.rstrip('\n') for line in doc]
        with open(result_file, 'r') as f_ans:
            doc = f_ans.readlines()
            results = [line.rstrip('\n') for line in doc]

        if not len(gold1) == len(gold2) == len(gold3) == len(queries) == len(results):
            raise Exception('数据长度不一致！')

        hit1 = 0
        hit3 = 0
        for res, g1, g2, g3 in zip(results, gold1, gold2, gold3):
            if res == g1 or res == g2 or res == g3:
                hit1 += 1
                hit3 += 1

    def __match(self, gold, ans_list, exact_match=False):
        # 如果答案是-1，说明不可回答，那么肯定和正确答案匹配不上
        if ans_list[0] == '-一':
            return False

        if exact_match:
            return gold in ans_list
        else:
            return self.__tfidf_sim_match(gold, ans_list)

    # 用tfidf-sim扩充标准答案集合
    def __tfidf_sim_match(self, query, ans_list, threshold=0.10):
        cut_query, _ = self.pre_processor.clean_cut_trim([query])  # 清洗query
        cut_ans_list, _ = self.pre_processor.clean_cut_trim(ans_list)  # 清洗ans_list

        ans_bow = [self.tfidf_dict.doc2bow(line) for line in cut_ans_list]  # 用ans_list做一个bag of words
        text_tfidf = self.tfidf_model[ans_bow]  # apply model
        sim_index = SparseMatrixSimilarity(text_tfidf, self.n_features)

        query_bow = [self.tfidf_dict.doc2bow(cut_query[0])]  # 用query做一个bag of words
        query_tfidf = self.tfidf_model[query_bow]  # 用tfidf model编码
        similarities = sim_index[query_tfidf][0]  # 算相似度

        sorted_scores = sorted(similarities, reverse=True)  # 将得分从大到小排序
        max_pos = np.argsort(similarities)[::-1]  # 从大到小排序，返回index(而不是真正的value)
        answers = self.__max_pos2answers(max_pos, ans_list)

        # 用QQ匹配的阈值过滤一遍结果
        sorted_scores, max_pos, answers, questions = \
            self.__filter_by_threshold(sorted_scores, max_pos, answers, [], threshold)

        if len(answers) > 0:
            return True
        else:
            return False

    # 根据max_pos从答案库里把真正的答案抽出来
    def __max_pos2answers(self, max_pos, uncut_answers):
        max_pos = max_pos.tolist()  # ndarray -> list
        answers = []
        for r in max_pos:
            answers.append(uncut_answers[r])
        return answers

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


if __name__ == "__main__":
    evaluator = Evaluate()
    evaluator.evaluate()

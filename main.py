# -*- coding: utf-8 -*-

import csv
import re
import argparse

from search_algorithms import *
from reader import Reader
from post_processor import PostProcessor

# 载入参数
parser = argparse.ArgumentParser()
# 使用的算法或模型
parser.add_argument('--alg', default='qq-match', type=str,
                    help='supported alg: bm25, qq-match, mix, tfidf, aver-embed, lm')
# 要不要去掉停用词
parser.add_argument('--trim_stop', default=False, type=bool, help='trim stopwords or not')
# 要不要加入长答案(每个小标题下所有内容的集合)
parser.add_argument('--long_ans', default=True, type=bool, help='chooses whether to use long answers')
args = parser.parse_args()


def search_answers(cleaned_in, uncut_in, cleaned_ans_json, cleaned_ans_txt):
    baseline_model = Baselines(cleaned_ans_json, cleaned_ans_txt)

    sorted_scores_list = []
    answers_list = []
    answers_index_list = []
    questions_list = []

    for i, (cut_query, query) in enumerate(zip(cleaned_in, uncut_in)):
        # 用不同算法搜索
        if method == 'bm25':
            sorted_scores, max_pos, answers = baseline_model.bm25(cut_query, baseline_model.cut_answers)
        elif method == 'qq-match':
            sorted_scores, max_pos, answers, questions = baseline_model.qq_match(cut_query)
            questions_list.append(questions)
        elif method == 'mix':
            sorted_scores, max_pos, answers, questions = baseline_model.qq_qa_mix(cut_query)
            questions_list.append(questions)
        elif method == 'tfidf-sim':
            sorted_scores, max_pos, answers = baseline_model.tfidf_sim(cut_query, baseline_model.cut_answers)
        # elif method == 'tfidf-dist':
        #     result = Baselines.tfidf_dist(cut_query, cleaned_ans_json)
        # elif method == 'tfidf':
        #     result = Baselines.tfidf(cut_query, cleaned_ans_txt)
        elif method == 'aver-embed':
            sorted_scores, max_pos, answers = baseline_model.aver_embed(cut_query)
        elif method == 'lm':
            sorted_scores, max_pos, answers = baseline_model.language_model(cut_query)
        else:
            raise Exception('尚未支持该搜索算法！')

        sorted_scores_list.append(sorted_scores)
        answers_list.append(answers)
        answers_index_list.append(max_pos)

        # 输出实时进度
        if i % 20 == 0:
            print('line ' + str(i) + ' processed')

    if method == 'mix':
        print('QQ count:', baseline_model.qq_count)
        print('QA count:', baseline_model.qa_count)
    if method == 'qq-match' or 'mix':
        post_processor.print_answers(questions_list, 'data/output_questions.csv')

    return answers_list, answers_index_list, sorted_scores_list


if __name__ == '__main__':
    # 文件名
    raw_docx1 = 'data/长沙机场知识库(没目录图片表格).docx'
    raw_docx2 = 'data/96566机场问询资料.docx'
    answers_txt = 'data/answers.txt'
    extra_txt = 'data/extra_answers.txt'
    cleaned_answers_json = 'data/cleaned_answers.json'
    cleaned_answers_txt = 'data/cleaned_answers.txt'
    cleaned_extra_txt = 'data/cleaned_extra.txt'
    cleaned_extra_json = 'data/cleaned_extra.json'
    long_answers_txt = 'data/long_answers.txt'
    long_answers_json = 'data/long_answers.json'
    input_txt = 'data/input.txt'
    output_csv = 'data/output.csv'
    stopword_txt = 'data/stopword.txt'

    reader = Reader(args, stopword_txt, input_txt, [raw_docx1, raw_docx2], answers_txt, extra_txt,
                    cleaned_answers_txt, cleaned_answers_json,
                    long_answers_txt, long_answers_json,
                    cleaned_extra_txt, cleaned_extra_json)  # 实例化一个Reader类
    post_processor = PostProcessor()

    # queries
    with open(input_txt, 'r') as f_input:
        input = f_input.readlines()
        input = [line.rstrip('\n') for line in input]
    cleaned_input, uncut_input = reader.clean_input(input)
    # 下一行代码只用运行一次
    reader.preprocess()  # 预处理数据

    # 清空输出文件
    with open(output_csv, 'w') as f_out:
        pass
    with open('data/output_questions.csv', 'w') as f_tmp:
        pass
    with open('data/scores.csv', 'w') as f_score:
        pass

    method = args.alg
    print('current algorithm:', method)  # 反馈当前使用的算法

    if args.long_ans:
        print('using long answers')
        answers_list, answer_idx_list, sorted_scores_list = \
            search_answers(cleaned_input, uncut_input, long_answers_json, long_answers_txt)
        if method != 'qq-match' and method != 'mix':
            answers_list = post_processor.clean_answers(list(answers_list), list(answer_idx_list), long_answers_txt)  # 清洗答案
        else:
            post_processor.print_answers(sorted_scores_list, 'data/scores.csv')
    else:
        print('NOT using long answers')
        answers_list, answer_idx_list, sorted_scores_list = \
            search_answers(cleaned_input, uncut_input, cleaned_answers_json, cleaned_answers_txt)
        if method != 'qq-match' and method != 'mix':
            answers_list = post_processor.clean_answers(list(answers_list), list(answer_idx_list), cleaned_answers_txt)  # 清洗答案
        else:
            post_processor.print_answers(sorted_scores_list, 'data/scores.csv')

    post_processor.print_answers(answers_list, output_csv)  # 打印答案

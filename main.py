# -*- coding: utf-8 -*-

import csv
import re
import argparse

from search_algorithms import *
from reader import Reader

# 载入参数
parser = argparse.ArgumentParser()
# 使用的算法或模型
parser.add_argument('--alg', default='qq-match', type=str,
                    help='supported alg: bm25, qq-match, tfidf, aver-embed, lm, bert and ernie')
# 要不要去掉停用词
parser.add_argument('--trim_stop', default=False, type=bool, help='trim stopwords or not')
# 要不要加入长答案(每个小标题下所有内容的集合)
parser.add_argument('--long_ans', default=True, type=bool, help='chooses whether to use long answers')
args = parser.parse_args()


def search_answers(cleaned_in, uncut_in, cleaned_ans_json, cleaned_ans_txt, word2vec_pkl,
                   queries_file='data/queries.json'):
    """

    Parameters
    ----------
    cleaned_in : list of list of str
        分词、去停用词后的输入
    uncut_in : list of str
        仅去停用词后的输入(没有分词)
    cleaned_ans_json
    cleaned_ans_txt

    Returns
    -------
    answers_list : list of string
        答案的列表，按相关度排序
    result : list of int
        答案在文档中的index的列表，按相关度排序
    """
    # 从文档中找出答案
    with open(cleaned_ans_txt, 'r') as f_ans_txt:
        text = f_ans_txt.readlines()

    # 读入queries
    with open(queries_file, 'r') as f_queries:
        queries = json.load(f_queries)

    deep_model = NeuralNetworks()
    baseline_model = Baselines(cleaned_ans_json, word2vec_pkl)

    answers_list = []
    answers_index_list = []
    questions_list = []

    for i, (cut_query, query) in enumerate(zip(cleaned_in, uncut_in)):
        # 用不同算法搜索
        if method == 'bm25':
            result = baseline_model.bm25(cut_query, baseline_model.cut_answers)
        elif method == 'qq-match':
            result = baseline_model.qq_match(cut_query)
        elif method == 'tfidf-sim':
            result = Baselines.tfidf_sim(cut_query, cleaned_ans_json)
        elif method == 'tfidf-dist':
            result = Baselines.tfidf_dist(cut_query, cleaned_ans_json)
        elif method == 'tfidf':
            result = Baselines.tfidf(cut_query, cleaned_ans_txt)
        elif method == 'aver-embed':
            result = baseline_model.aver_embed(cut_query)
        elif method == 'lm':
            result = baseline_model.language_model(cut_query)
        elif method == 'bert':
            result = deep_model.bert_search(query, cleaned_ans_txt)
        elif method == 'ernie':
            result = deep_model.ernie_search(query, cleaned_ans_txt)
        else:
            raise Exception('尚未支持该搜索算法！')

        # ndarray -> list
        result = result.tolist()

        # 特殊处理qq-match的情况
        if method == 'qq-match':
            answers = []
            questions = []

            # 检查top answer是不是正确答案
            idx = result[0]
            if baseline_model.base_questions[idx]['sentence'] == queries[i]['sentence']:
                answers.append(1)
            else:
                answers.append(0)

            for r in result:
                if r != -1:
                    answers.append(baseline_model.base_questions[r]['sentence'])
                    questions.append(baseline_model.base_questions[r]['question'])
                else:
                    answers.append('-')  # 丢弃该回答
                    questions.append('-')
            answers_list.append(answers)
            questions_list.append(questions)

        else:
            answers = []
            for r in result:
                if r != -1:
                    answers.append(text[r].rstrip())
                else:
                    answers.append('-')  # 丢弃该回答
            answers_list.append(answers)
            answers_index_list.append(result)

        # 输出实时进度
        if i % 20 == 0:
            print('line ' + str(i) + ' processed')

    if method == 'qq-match':
        print_answers(questions_list, 'data/output_questions.csv')
    return answers_list, answers_index_list


def clean_answers(ans_in, ans_idx_in, cleaned_ans_txt):
    # 入参检查
    if len(ans_in) != len(ans_idx_in):
        raise Exception('答案列表和答案索引列表长度不一致！')

    # 载入答案文档
    with open(cleaned_ans_txt, 'r') as f_ans_txt:
        text = f_ans_txt.readlines()

    ret = []
    for answers, answer_indexes in zip(ans_in, ans_idx_in):
        cleaned_ans = []
        for ans, ans_idx in zip(answers, answer_indexes):
            if re.match(r'\d+\.\d+', ans) is not None:
                # 如果答案是二级标题，那么就取下一个段落(下个段落有可能是三级标题)
                next_ans = text[ans_idx + 1].rstrip()
                next_idx = ans_idx + 1
                # print('标题：' + ans + ' -> 内容：' + next_ans)
                # 再判断一下是不是三级标题
                if re.match(r'\d+\.\d+\.\d+', next_ans) is not None:
                    real_ans = text[next_idx + 1].rstrip()
                    cleaned_ans.append(real_ans)
                    # print('标题：' + next_ans + ' -> 内容：' + real_ans)
                else:
                    cleaned_ans.append(next_ans)

            elif re.match(r'\d+\.\d+\.\d+', ans) is not None:
                # 如果答案是三级标题，那么下一个段落就是真正的答案
                real_ans = text[ans_idx + 1].rstrip()
                cleaned_ans.append(real_ans)
                # print('标题：' + ans + ' -> 内容：' + real_ans)

            elif re.match('.+([？?])$', ans) is not None:
                # 如果答案以问号结尾，那么就跳过
                continue

            else:
                # 如果以上条件都不符合，那么这个答案正常保留
                cleaned_ans.append(ans)

        ret.append(cleaned_ans)
    return ret


def print_answers(ans_list, output, print2file=True, n_result=3):
    """
        Parameters
        ----------
        ans_list
        output
        print2file
        n_result : int
            返回前n个结果
    """
    if print2file:
        with open(output, 'a') as f_out:
            writer = csv.writer(f_out, lineterminator='\n')
            for answers in ans_list:
                writer.writerow(answers[:n_result])
    else:
        for answers in ans_list:
            print('搜索结果(前' + str(n_result) + '个)：')
            print(' '.join(answers))


if __name__ == '__main__':
    # 文件名
    raw_docx = 'data/长沙机场知识库(没目录图片表格).docx'
    answers_txt = 'data/answers.txt'
    extra_txt = 'data/extra_answers.txt'
    cleaned_answers_json = 'data/cleaned_answers.json'
    cleaned_answers_txt = 'data/cleaned_answers.txt'
    cleaned_extra_txt = 'data/cleaned_extra.txt'
    cleaned_extra_json = 'data/cleaned_extra.json'
    long_answers_txt = 'data/long_answers.txt'
    long_answers_json = 'data/long_answers.json'
    word2vec_pickle = 'data/word2vec.pickle'
    input_txt = 'data/input.txt'
    output_csv = 'data/output.csv'
    stopword_txt = 'data/stopword.txt'

    reader = Reader(args, stopword_txt, input_txt, raw_docx, answers_txt, extra_txt,
                    cleaned_answers_txt, cleaned_answers_json,
                    long_answers_txt, long_answers_json,
                    cleaned_extra_txt, cleaned_extra_json)  # 实例化一个Reader类
    cleaned_input, uncut_input = reader.clean_input()
    # 下一行代码只用运行一次
    reader.preprocess()  # 预处理数据

    # 清空输出文件
    with open(output_csv, 'w') as f_out:
        pass
    with open('data/output_questions.csv', 'w') as f_tmp:
        pass

    method = args.alg
    print('current algorithm:', method)  # 反馈当前使用的算法

    if args.long_ans:
        print('using long answers')
        answers_list, answer_idx_list = search_answers(cleaned_input, uncut_input,
                                                       long_answers_json, long_answers_txt, word2vec_pickle)
        if method != 'qq-match':
            answers_list = clean_answers(list(answers_list), list(answer_idx_list), long_answers_txt)  # 清洗答案
    else:
        print('NOT using long answers')
        answers_list, answer_idx_list = search_answers(cleaned_input, uncut_input,
                                                       cleaned_answers_json, cleaned_answers_txt, word2vec_pickle)
        if method != 'qq-match':
            answers_list = clean_answers(list(answers_list), list(answer_idx_list), cleaned_answers_txt)  # 清洗答案

    if method == 'qq-match':
        # 多打印一列，因为第一列是系统给出的hit1
        print_answers(answers_list, output_csv, n_result=4)  # 打印答案
    else:
        print_answers(answers_list, output_csv)  # 打印答案

# -*- coding: utf-8 -*-

import docx
import json
import jieba
import csv
import re

from search_algorithms import *


# 载入word文档并转成txt文件
def read_doc(input, output):
    f_raw = docx.Document(input)
    # 按段落分割，并写到一个txt文件里
    with open(output, 'w') as f_answers:
        for para in f_raw.paragraphs:
            f_answers.write(para.text)
            f_answers.write('\n')


# 清洗数据
def clean_txt(input, output_json, output_txt):
    cleaned_answers_json = []  # 清洗过后的数据(json格式)
    with open(input, 'r') as f_in, open(output_txt, 'w') as f_out_txt:
        for line in f_in:
            # 去掉行首空格
            line = line.lstrip()
            # 去掉空行
            if line == '':
                continue
            # 分词
            line_json = [w for w in jieba.cut(line)]
            # 以json和txt两种格式保存数据
            cleaned_answers_json.append(line_json)
            f_out_txt.write(line)
    with open(output_json, 'w') as f_out_json:
        json.dump(obj=cleaned_answers_json, fp=f_out_json, ensure_ascii=False)


def search_answers(input, cleaned_ans_json, cleaned_ans_txt):
    """

    Parameters
    ----------
    input
    cleaned_ans_json
    cleaned_ans_txt

    Returns
    -------
    answers_list : list of string
        答案的列表，按相关度排序
    result : list of int
        答案在文档中的index的列表，按相关度排序
    """
    answers_list = []
    answers_index_list = []
    with open(input, 'r') as f_in:
        for line in f_in:
            cut_query = [w for w in jieba.cut(line.rstrip())]  # 对query进行分词
            query = line.rstrip()

            # 用不同算法搜索
            neural_nets = NeuralNetworks()
            if method == 'bm25':
                result = Baselines.bm25(cut_query, cleaned_ans_json)
            elif method == 'tfidf-sim':
                result = Baselines.tfidf_sim(cut_query, cleaned_ans_json)
            elif method == 'tfidf-dist':
                result = Baselines.tfidf_dist(cut_query, cleaned_ans_json)
            elif method == 'tfidf':
                result = Baselines.tfidf(cut_query, cleaned_ans_txt)
            elif method == 'bert':
                result = neural_nets.bert_search(query, cleaned_ans_txt)
            elif method == 'ernie':
                result = neural_nets.ernie_search(query, cleaned_ans_txt)
            else:
                raise Exception('尚未支持该搜索算法！')

            # ndarray -> list
            result = result.tolist()

            # 从文档中找出答案
            with open(cleaned_ans_txt, 'r') as f_ans_txt:
                text = f_ans_txt.readlines()
            answers = [text[r].rstrip() for r in result]
            answers_list.append(answers)
            answers_index_list.append(result)
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
            if re.match(r'\d\.\d.\d', ans) is not None:
                # 如果答案是小标题，那么下一个段落就是真正的答案
                real_ans = text[ans_idx + 1].rstrip()
                cleaned_ans.append(real_ans)
                print('标题：' + ans + ' -> 内容：' + real_ans)
            elif re.match('.+([？?])$', ans) is not None:
                # 如果答案以问号结尾，那么就跳过
                continue
            else:
                # 如果以上条件都不符合，那么这个答案正常保留
                cleaned_ans.append(ans)
        ret.append(cleaned_ans)
    return ret


def print_answers(ans_list, output, print2file=True, n_result=5):
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
    cleaned_answers_json = 'data/cleaned_answers.json'
    cleaned_answers_txt = 'data/cleaned_answers.txt'
    input_txt = 'data/input.txt'
    output_csv = 'data/output.csv'

    supported_alg = ['bm25', 'tfidf-dist', 'tfidf-sim', 'tfidf', 'bert', 'ernie']  # 支持的算法

    # 以下两行只用运行一次
    read_doc(raw_docx, answers_txt)  # 读文档
    clean_txt(answers_txt, cleaned_answers_json, cleaned_answers_txt)  # 清洗数据

    # 清空输出文件
    with open(output_csv, 'w') as f_out:
        pass

    # 从控制台输入将要使用的算法
    print('当前支持的算法：', supported_alg)
    method = ''
    while method not in supported_alg:
        method = input('请输入搜索算法：')

    answers_list, answer_idx_list = search_answers(input_txt, cleaned_answers_json, cleaned_answers_txt)  # 回答问题
    answers_list = clean_answers(answers_list, answer_idx_list, cleaned_answers_txt)  # 清洗答案
    print_answers(answers_list, output_csv)  # 打印答案

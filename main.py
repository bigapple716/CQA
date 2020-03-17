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
        print_answers(questions_list, 'data/output_questions.csv')

    return answers_list, answers_index_list, sorted_scores_list


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
            answers_list = clean_answers(list(answers_list), list(answer_idx_list), long_answers_txt)  # 清洗答案
        else:
            print_answers(sorted_scores_list, 'data/scores.csv')
    else:
        print('NOT using long answers')
        answers_list, answer_idx_list, sorted_scores_list = \
            search_answers(cleaned_input, uncut_input, cleaned_answers_json, cleaned_answers_txt)
        if method != 'qq-match' and method != 'mix':
            answers_list = clean_answers(list(answers_list), list(answer_idx_list), cleaned_answers_txt)  # 清洗答案
        else:
            print_answers(sorted_scores_list, 'data/scores.csv')

    print_answers(answers_list, output_csv)  # 打印答案

# -*- coding: utf-8 -*-

import json
import csv
from textqa.model.CQA.file_pool import FilePool
from textqa.model.CQA.utils import Utils


# 自动化测试
class Evaluate:
    def __init__(self):
        # 读入问题
        with open(FilePool.input_txt, 'r') as f_q:
            doc = f_q.readlines()
            self.queries = Utils.clean_text(doc)  # 清洗输入
        # 读入top1和top3的回答
        with open(FilePool.output_csv, 'r') as f_res:
            reader = csv.reader(f_res, lineterminator='\n')
            self.top1_answers = []
            self.top3_answers = []
            for row in reader:
                self.top1_answers.append([row[0]])
                self.top3_answers.append(row)
        # 读入问答对
        with open(FilePool.qa_file, 'r') as f_qa:
            self.qa = json.load(f_qa)
            # 清洗问题
            for dict in self.qa:
                dict['question'] = Utils.clean_line(dict['question'])

        # 校验问题数量和答案数量是否相等
        if len(self.queries) != len(self.top1_answers):
            raise Exception('问题和答案数量不相等！')

    def evaluate(self, hit1=True):
        # 选择统计top1还是top3
        if hit1:
            answers = self.top1_answers
        else:
            answers = self.top3_answers

        hit = 0
        found = 0

        for query, ans_list in zip(self.queries, answers):
            for dict in self.qa:  # 遍历QA库
                if query == dict['question']:
                    found += 1
                    # 在QA库里找到这个问题了，看看有没有gold在top1/top3里面的
                    for gold in dict['sentence']:
                        if gold in ans_list:
                            hit += 1
                            break

        # 输出结果
        if hit1:
            print('hit1:', hit / len(answers))
        else:
            print('hit3:', hit / len(answers))
        print(found)

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


if __name__ == "__main__":
    evaluator = Evaluate()
    evaluator.evaluate_from_txt()

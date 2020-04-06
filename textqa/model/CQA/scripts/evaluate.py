# -*- coding: utf-8 -*-

import json
import csv
from textqa.model.CQA.file_pool import FilePool


# 自动化测试
class Evaluate:
    def __init__(self):
        # 读入问题
        with open(FilePool.input_txt, 'r') as f_q:
            doc = f_q.readlines()
            self.queries = [line.rstrip('\n') for line in doc]
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

        # 校验问题数量和答案数量是否相等
        if len(self.queries) != len(self.top1_answers):
            raise Exception('问题和答案数量不相等！')

    # 这个类里的主方法
    def evaluate(self, hit1=True):
        # 选择统计top1还是top3
        if hit1:
            answers = self.top1_answers
        else:
            answers = self.top3_answers

        hit = 0

        for query, ans_list in zip(self.queries, answers):
            for dict in self.qa:  # 遍历QA库
                if query == dict['question']:
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


if __name__ == "__main__":
    evaluator = Evaluate()
    evaluator.evaluate()

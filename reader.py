# -*- coding: utf-8 -*-

import docx
import jieba
import json

from utils import Utils


class Reader:
    def __init__(self, in_docx, ans_txt, cleaned_ans_json, cleaned_ans_txt, long_ans_txt, long_ans_json):
        self.raw_docx = in_docx
        self.answers_txt = ans_txt
        self.cleaned_answers_json = cleaned_ans_json
        self.cleaned_answers_txt = cleaned_ans_txt
        self.long_answers_txt = long_ans_txt
        self.long_answers_json = long_ans_json

    # 预处理数据(下面方法的集合)
    def preprocess(self):
        self.read_doc()
        self.clean_txt()
        self.merge_add()

    # 载入word文档并转成txt文件
    def read_doc(self):
        f_raw = docx.Document(self.raw_docx)
        # 按段落分割，并写到一个txt文件里
        with open(self.answers_txt, 'w') as f_answers:
            for para in f_raw.paragraphs:
                f_answers.write(para.text)
                f_answers.write('\n')

    # 清洗数据
    def clean_txt(self):
        cleaned_json = []  # 清洗过后的数据(json格式)
        with open(self.answers_txt, 'r') as f_in, open(self.cleaned_answers_txt, 'w') as f_out_txt:
            for line in f_in:
                # 去掉行首空格
                line = line.lstrip()
                # 去掉空行
                if line == '':
                    continue
                # 分词
                line_json = [w for w in jieba.cut(line)]
                # 以json和txt两种格式保存数据
                cleaned_json.append(line_json)
                f_out_txt.write(line)
        with open(self.cleaned_answers_json, 'w', encoding='utf-8') as f_out_json:
            json.dump(obj=cleaned_json, fp=f_out_json, ensure_ascii=False)

    # 把每个小标题下面的所有段落合成一个段落 & 加入到答案库中
    def merge_add(self):
        answers = []

        # 把清洗过的答案读进来
        with open(self.cleaned_answers_txt, 'r') as f_in:
            lines = f_in.readlines()

        long_ans = ''  # 用来存长答案的字符串
        short_answers = []  # 原来的短答案

        l = 0  # loop var
        while l < len(lines):
            # 如果这一行是标题
            if Utils.is_heading(lines[l]):
                # 如果ans非空，那么说明上一行是正文
                if long_ans != '':
                    # 把长答案存档并清空
                    answers.append(long_ans)
                    long_ans = ''
                    # 把短答案存档并清空
                    answers += short_answers
                    short_answers = []
                answers.append(lines[l].rstrip())  # 标题照搬到答案库里就完事了
            # 如果这一行是正文
            else:
                short_answers.append(lines[l].rstrip())  # 把这一行暂存到短答案表里
                long_ans += lines[l].rstrip()  # 把这一行加到ans后面就行
            l = l + 1

        # 如果ans非空，那么说明整篇文档最后一行是正文
        if long_ans != '':
            # 把长答案存档
            answers.append(long_ans)
            # 把短答案存档
            answers += short_answers

        # 将答案库存为txt格式
        with open(self.long_answers_txt, 'w') as f_out_txt:
            for line in answers:
                f_out_txt.write(line + '\n')

        # 将答案库存为json格式
        answers_json = []  # json格式答案库
        for line in answers:
            # 分词
            line_json = [w for w in jieba.cut(line)]
            answers_json.append(line_json)
        with open(self.long_answers_json, 'w') as f_out_json:
            json.dump(obj=answers_json, fp=f_out_json, ensure_ascii=False)

# -*- coding: utf-8 -*-

import docx
import jieba
import json


class Reader:
    def __init__(self, in_docx, ans_txt, cleaned_ans_json, cleaned_ans_txt):
        self.raw_docx = in_docx
        self.answers_txt = ans_txt
        self.cleaned_answers_json = cleaned_ans_json
        self.cleaned_answers_txt = cleaned_ans_txt

    # 预处理数据(下面方法的集合)
    def preprocess(self):
        self.read_doc()
        self.clean_txt()
        self.merge()

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

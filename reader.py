# -*- coding: utf-8 -*-

import docx
import jieba
import json
from utils import Utils
from file_pool import FilePool


class Reader:
    def __init__(self, trim_stop):
        self.trim_stop = trim_stop

        # 停用词表
        with open(FilePool.stopword_txt, 'r') as f_stopword:
            doc = f_stopword.readlines()
        self.stopwords = [line.rstrip('\n') for line in doc]

        # queries
        with open(FilePool.input_txt, 'r') as f_input:
            doc = f_input.readlines()
        self.input = [line.rstrip('\n') for line in doc]

        self.raw_docx_list = FilePool.docx_list
        self.answers_txt = FilePool.answers_txt
        self.extra_txt = FilePool.extra_txt
        self.cleaned_answers_txt = FilePool.cleaned_answers_txt
        self.cleaned_answers_json = FilePool.cleaned_answers_json
        self.cleaned_extra_txt = FilePool.cleaned_extra_txt
        self.cleaned_extra_json = FilePool.cleaned_extra_json
        self.long_answers_txt = FilePool.long_answers_txt
        self.long_answers_json = FilePool.long_answers_json

    # 输入去停用词
    def clean_input(self, text):
        """

        Returns
        -------
        cleaned : list of list of str
            分词、去停用词后的输入
        uncut : list of str
            仅去停用词后的输入(没有分词)
        """
        cleaned = []
        uncut = []
        for line in text:
            line = Utils.str2cn(line)  # 阿拉伯数字转中文
            cut_line = [w for w in jieba.cut(line)]  # 对query进行分词
            # 去停用词
            if self.trim_stop:
                trim_line = [w for w in cut_line if w not in self.stopwords]
            else:
                trim_line = cut_line

            uncut_line = ''.join(trim_line)
            cleaned.append(trim_line)
            uncut.append(uncut_line)

        if self.trim_stop:
            print('stop words trimmed')
        else:
            print('stop words NOT trimmed')
        return cleaned, uncut

    # 预处理数据(下面方法的集合)
    def preprocess(self):
        self.__read_docs()
        self.__clean_txt(self.answers_txt, self.cleaned_answers_txt, self.cleaned_answers_json)
        self.__clean_txt(self.extra_txt, self.cleaned_extra_txt, self.cleaned_extra_json)
        self.__merge_add()
        self.__add_extra()

    '''以下均为私有方法'''

    # 载入word文档并转成txt文件
    def __read_docs(self):
        with open(self.answers_txt, 'w') as f_answers:
            for raw_docx in self.raw_docx_list:
                # read .doc file
                f_raw = docx.Document(raw_docx)
                # 按段落分割，并写到一个txt文件里
                for para in f_raw.paragraphs:
                    f_answers.write(para.text)
                    f_answers.write('\n')

    # 清洗数据
    def __clean_txt(self, ans_file, cleaned_txt_file, cleaned_json_file):
        cleaned_json = []  # 清洗过后的数据(json格式)
        with open(ans_file, 'r') as f_in, open(cleaned_txt_file, 'w') as f_out_txt:
            for line in f_in:
                # 去掉行首空格
                line = line.lstrip()
                # 去掉空行
                if line == '':
                    continue
                line = line.replace('\t', '')  # 去掉\t
                line = Utils.full2half(line)  # 全角转半角
                line = Utils.str2cn(line)  # 阿拉伯数字转中文
                # 分词
                line_json = [w for w in jieba.cut(line.rstrip())]
                # 以json和txt两种格式保存数据
                cleaned_json.append(line_json)
                f_out_txt.write(line)
        with open(cleaned_json_file, 'w') as f_out_json:
            json.dump(obj=cleaned_json, fp=f_out_json, ensure_ascii=False)

    # 把每个小标题下面的所有段落合成一个段落 & 加入到答案库中
    def __merge_add(self):
        answers = []

        # 把清洗过的答案读进来
        with open(self.cleaned_answers_txt, 'r') as f_in:
            lines = f_in.readlines()

        long_ans = ''  # 用来存长答案的字符串
        short_answers = []  # 原来的短答案

        idx = 0  # loop var
        while idx < len(lines):
            # 如果这一行是标题
            if Utils.is_heading(lines[idx]):
                # 如果ans非空，那么说明上一行是正文
                if long_ans != '':
                    # 把长答案存档并清空
                    answers.append(long_ans)
                    long_ans = ''
                    # 把短答案存档并清空
                    answers += short_answers
                    short_answers = []
                answers.append(lines[idx].rstrip())  # 标题照搬到答案库里就完事了
            # 如果这一行是正文
            else:
                short_answers.append(lines[idx].rstrip())  # 把这一行暂存到短答案表里
                long_ans += lines[idx].rstrip()  # 把这一行加到ans后面就行
            idx = idx + 1

        # 如果ans非空，那么说明整篇文档最后一行是正文
        if long_ans != '':
            # 把长答案存档
            answers.append(long_ans)
            # 把短答案存档
            answers += short_answers

        answers = self.__remove_dup(answers)

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

    # 把补充答案合并到cleaned_answers和long_answers里面
    def __add_extra(self):
        # read extra
        with open(self.cleaned_extra_txt, 'r') as f_cleaned_extra_txt:
            cleaned_extra_txt = f_cleaned_extra_txt.readlines()
        # write
        with open(self.cleaned_answers_txt, 'a') as f_cleaned_ans_txt:
            f_cleaned_ans_txt.writelines(cleaned_extra_txt)
        # write long version
        with open(self.long_answers_txt, 'a') as f_long_ans_txt:
            f_long_ans_txt.writelines(cleaned_extra_txt)

        # read extra
        with open(self.cleaned_extra_json, 'r') as f_cleaned_extra_json:
            cleaned_extra_json = json.load(f_cleaned_extra_json)
        # write
        with open(self.cleaned_answers_json, 'r') as f_cleaned_ans_json:
            cleaned_ans_json = json.load(f_cleaned_ans_json)
            merged_json = cleaned_ans_json + cleaned_extra_json
        with open(self.cleaned_answers_json, 'w') as f_cleaned_ans_json:
            json.dump(obj=merged_json, fp=f_cleaned_ans_json, ensure_ascii=False)
        # write long version
        with open(self.long_answers_json, 'r') as f_long_ans_json:
            cleaned_ans_json = json.load(f_long_ans_json)
            merged_json = cleaned_ans_json + cleaned_extra_json
        with open(self.long_answers_json, 'w') as f_long_ans_json:
            json.dump(obj=merged_json, fp=f_long_ans_json, ensure_ascii=False)

    # 给list of str去重
    def __remove_dup(self, list_in):
        return list(dict.fromkeys(list_in))

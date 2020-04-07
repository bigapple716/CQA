# -*- coding: utf-8 -*-

import docx
import jieba
import json
import pickle
from textqa.model.CQA.utils import Utils
from textqa.model.CQA.file_pool import FilePool
from textqa.model.CQA import args


class Reader:
    # 预处理数据(下面方法的集合)
    def preprocess(self):
        self.__read_keywords()  # 读入各类的关键词
        self.__clean_txt(FilePool.raw_small_answers_txt, FilePool.small_answers_txt, FilePool.small_answers_json)
        self.__make_base_que_list()  # 制作base_ques_list

        # 以下方法有先后依赖关系
        self.__read_docs()
        self.__clean_txt(FilePool.answers_txt, FilePool.cleaned_answers_txt, FilePool.cleaned_answers_json)
        self.__clean_txt(FilePool.extra_txt, FilePool.cleaned_extra_txt, FilePool.cleaned_extra_json)
        self.__merge_add()
        self.__add_extra()

    '''以下均为私有方法'''

    # 载入word文档并转成txt文件
    def __read_docs(self):
        with open(FilePool.answers_txt, 'w') as f_answers:
            for raw_docx in FilePool.docx_list:
                # read .doc file
                f_raw = docx.Document(raw_docx)
                # 按段落分割，并写到一个txt文件里
                for para in f_raw.paragraphs:
                    f_answers.write(para.text)
                    f_answers.write('\n')

    # 清洗数据
    def __clean_txt(self, ans_file, cleaned_txt_file, cleaned_json_file):
        cleaned_json = []  # 清洗过后的数据(json格式)
        cleaned_txt = []  # 清洗过后的数据(txt格式)

        with open(ans_file, 'r') as f_in:
            for line in f_in:
                line = line.lstrip()  # 去掉行首空格
                line = line.rstrip()  # 去掉行尾换行符
                if line == '':  # 去掉空行
                    continue
                line = line.replace('\t', '')  # 去掉\t
                line = Utils.full2half(line)  # 全角转半角
                line = Utils.str2cn(line)  # 阿拉伯数字转中文
                cleaned_txt.append(line)  # 加入到cleaned_txt这个列表里
                line_json = [w for w in jieba.cut(line)]  # 分词
                cleaned_json.append(line_json)

        # 写进文件
        with open(cleaned_txt_file, 'w') as f_out_txt:
            for line in cleaned_txt:
                f_out_txt.write(line + '\n')
        with open(cleaned_json_file, 'w') as f_out_json:
            json.dump(obj=cleaned_json, fp=f_out_json, ensure_ascii=False)

    # 把每个小标题下面的所有段落合成一个段落 & 加入到答案库中
    def __merge_add(self):
        long_answers = []

        # 把清洗过的答案读进来
        with open(FilePool.cleaned_answers_txt, 'r') as f_in:
            lines = f_in.readlines()
            lines = [line.rstrip() for line in lines]

        one_long_ans = ''  # 用来存1条长答案的字符串
        short_answers = []  # 原来的短答案

        idx = 0  # loop var
        while idx < len(lines):
            # 如果这一行是标题
            if Utils.is_heading(lines[idx]):
                # 如果ans非空，那么说明上一行是正文
                if one_long_ans != '':
                    # 把长答案存档并清空
                    long_answers.append(one_long_ans)
                    one_long_ans = ''
                    # 把短答案存档并清空
                    long_answers += short_answers
                    short_answers = []
                long_answers.append(lines[idx])  # 标题照搬到答案库里就完事了
            # 如果这一行是正文
            else:
                short_answers.append(lines[idx])  # 把这一行暂存到短答案表里
                one_long_ans += lines[idx]  # 把这一行加到ans后面就行
            idx += 1

        # 如果ans非空，那么说明整篇文档最后一行是正文
        if one_long_ans != '':
            # 把长答案存档
            long_answers.append(one_long_ans)
            # 把短答案存档
            long_answers += short_answers

        long_answers = self.__remove_dup(long_answers)  # 去重

        # 将长答案存为txt格式
        with open(FilePool.long_answers_txt, 'w') as f_out_txt:
            for line in long_answers:
                f_out_txt.write(line + '\n')

        # 将答案库存为json格式
        long_answers_json = []  # json格式答案库
        for line in long_answers:
            # 分词
            line_json = [w for w in jieba.cut(line)]
            long_answers_json.append(line_json)
        with open(FilePool.long_answers_json, 'w') as f_out_json:
            json.dump(obj=long_answers_json, fp=f_out_json, ensure_ascii=False)

    # 把补充答案合并到cleaned_answers和long_answers里面
    def __add_extra(self):
        # read extra
        with open(FilePool.cleaned_extra_txt, 'r') as f_cleaned_extra_txt:
            cleaned_extra_txt = f_cleaned_extra_txt.readlines()
        # write
        with open(FilePool.cleaned_answers_txt, 'a') as f_cleaned_ans_txt:
            f_cleaned_ans_txt.writelines(cleaned_extra_txt)
        # write long version
        with open(FilePool.long_answers_txt, 'a') as f_long_ans_txt:
            f_long_ans_txt.writelines(cleaned_extra_txt)

        # read extra
        with open(FilePool.cleaned_extra_json, 'r') as f_cleaned_extra_json:
            cleaned_extra_json = json.load(f_cleaned_extra_json)
        # write
        with open(FilePool.cleaned_answers_json, 'r') as f_cleaned_ans_json:
            cleaned_ans_json = json.load(f_cleaned_ans_json)
            merged_json = cleaned_ans_json + cleaned_extra_json
        with open(FilePool.cleaned_answers_json, 'w') as f_cleaned_ans_json:
            json.dump(obj=merged_json, fp=f_cleaned_ans_json, ensure_ascii=False)
        # write long version
        with open(FilePool.long_answers_json, 'r') as f_long_ans_json:
            cleaned_ans_json = json.load(f_long_ans_json)
            merged_json = cleaned_ans_json + cleaned_extra_json
        with open(FilePool.long_answers_json, 'w') as f_long_ans_json:
            json.dump(obj=merged_json, fp=f_long_ans_json, ensure_ascii=False)

    # 去重
    def __remove_dup(self, list_in):
        return list(dict.fromkeys(list_in))

    # 读入各类的关键词
    def __read_keywords(self):
        keyword_database = []  # 关键词库
        for keyword_file in FilePool.keyword_list:
            with open(keyword_file, 'r') as f_kw:
                lines = f_kw.readlines()
                lines = [line.rstrip() for line in lines]
                keywords = lines[1].split(' ')
                uncut_answers = lines[2:]
                cut_answers = Utils.cut_text(uncut_answers)
                dict = {
                    'class': lines[0],
                    'keywords': keywords,
                    'uncut_answers': uncut_answers,
                    'cut_answers': cut_answers
                }
                keyword_database.append(dict)
        # 写到json里
        with open(FilePool.keyword_database_json, 'w') as f_kwdb:
            json.dump(obj=keyword_database, fp=f_kwdb, ensure_ascii=False)

    def __make_base_que_list(self):
        with open(FilePool.qa_file, 'r') as f_base_ques:
            base_questions = json.load(f_base_ques)

        # 把被匹配的问题分词，制作一个纯list
        base_ques_list = []
        for base_ques in base_questions:
            line = [w for w in jieba.cut(base_ques['question'])]
            base_ques_list.append(line)

        with open(FilePool.base_ques_list_file, 'w') as f_base_ques_list:
            json.dump(obj=base_ques_list, fp=f_base_ques_list, ensure_ascii=False)


# for test purpose
if __name__ == '__main__':
    reader = Reader()

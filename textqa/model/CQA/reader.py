# -*- coding: utf-8 -*-

import docx
import jieba
import json
import pickle
from textqa.model.CQA.utils import Utils
from textqa.model.CQA.file_pool import FilePool
from textqa.model.CQA import args


class Reader:
    def __init__(self):
        # 停用词表
        with open(FilePool.stopword_txt, 'r') as f_stopword:
            doc = f_stopword.readlines()
        self.stopwords = [line.rstrip('\n') for line in doc]

    # 清洗text
    def clean_input(self, text):
        """
        Returns
        -------
        cleaned : list of list of str
            清洗、分词后的输入
        uncut : list of str
            仅清洗后的输入(没有分词)
        """
        cleaned = []
        uncut = []
        for line in text:
            line = Utils.str2cn(line)  # 阿拉伯数字转中文
            cut_line = [w for w in jieba.cut(line)]  # 对query进行分词
            # 去停用词
            if args.trim_stop:
                trim_line = [w for w in cut_line if w not in self.stopwords]
            else:
                trim_line = cut_line

            uncut_line = ''.join(trim_line)
            cleaned.append(trim_line)
            uncut.append(uncut_line)
        return cleaned, uncut

    # 预处理数据(下面方法的集合)
    def preprocess(self):
        self.__read_keywords()  # 读入各类的关键词
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
        long_answers = []
        keyword_of_long_ans = {}

        # 把清洗过的答案读进来
        with open(FilePool.cleaned_answers_txt, 'r') as f_in:
            lines = f_in.readlines()
            lines = [line.rstrip() for line in lines]

        one_long_ans = ''  # 用来存1条长答案的字符串
        short_answers = []  # 原来的短答案
        current_keyword = '其他'

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
                current_keyword = lines[idx]  # 设置当前的关键词
                keyword_of_long_ans[current_keyword] = set()  # 在字典里，为这个关键词建一个空集合
            # 如果这一行是正文
            else:
                short_answers.append(lines[idx])  # 把这一行暂存到短答案表里
                one_long_ans += lines[idx]  # 把这一行加到ans后面就行
                keyword_of_long_ans[current_keyword].add(lines[idx])  # 把这句正文加到当前关键词对应的集合里
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

        # 把关键词to答案的字典保存为pickle格式
        with open(FilePool.keyword_of_answer, 'wb') as f_pkl:
            pickle.dump(keyword_of_long_ans, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)

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
                dict = {'class': lines[0], 'keywords': keywords, 'answers': lines[2:]}
                keyword_database.append(dict)
        # 写到json里
        with open(FilePool.keyword_database_json, 'w') as f_kwdb:
            json.dump(obj=keyword_database, fp=f_kwdb, ensure_ascii=False)


# for test purpose
if __name__ == '__main__':
    reader = Reader()

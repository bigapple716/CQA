# -*- coding: utf-8 -*-

from cn2an import an2cn
import re
from gensim.models import KeyedVectors
import pickle
import docx
import jieba
from textqa.model.CQA.file_pool import FilePool


# 静态工具类
class Utils:
    # 判断一个句子是不是标题
    @staticmethod
    def is_heading(sentence):
        # 长沙文档的标题
        # 节标题
        if re.match(r'^第.{1,2}节', sentence) is not None:
            # print(sentence)
            return True
        # 章标题
        elif re.match(r'^第.{1,2}章', sentence) is not None:
            # print(sentence)
            return True
        # 2级标题
        elif re.match(r'^\d+\.\d+[^a-zA-Z]', sentence) is not None:
            # print(sentence)
            return True
        # 3级标题
        elif re.match(r'^\d+\.\d+\.\d+', sentence) is not None:
            # print(sentence)
            return True
        elif len(sentence) <= 7 and re.match(r'^[一二三四五六七八九]、', sentence) is not None:
            # print(sentence)
            return True
        # 96566机场文档的标题
        elif re.match(r'^\(.\)、', sentence) is not None:
            # print(sentence)
            return True
        else:
            # 肯定不是标题
            return False

    # 从字符串里提取阿拉伯数字并转成中文
    @staticmethod
    def str2cn(str):
        pattern = r'(?<![\d.])[0-9]+(?![\d.])'
        return re.sub(pattern, Utils.__int2cn, str, count=0)

    # 全角 -> 半角
    @staticmethod
    def full2half(full_str):
        half_str = ''
        for uchar in full_str:
            code = ord(uchar)
            if code == 12288:  # 全角空格直接转换
                code = 32
            elif 65281 <= code <= 65374:  # 全角字符（除空格）根据关系转化
                code -= 65248
            half_str += chr(code)
        return half_str

    # 去掉小标题(交差时才会用到)
    @staticmethod
    def delete_titles(file_in, file_out):
        with open(file_in, 'r') as f_in:
            text = f_in.readlines()
        with open(file_out, 'w') as f_out:
            for line in text:
                if not Utils.is_heading(line):
                    f_out.write(line)

    # 统计文件里平均行长度
    @staticmethod
    def aver_len(input_file):
        with open(input_file, 'r') as f_in:
            text = f_in.readlines()

        total_len = 0
        for line in text:
            total_len += len(line.rstrip())
        return total_len / len(text)

    # load word2vec text file and dump it to pickle
    @staticmethod
    def load_embed(word2vec_file):
        word2vec = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
        word2vec.save_word2vec_format(FilePool.word2vec_bin, binary=True)
        with open(FilePool.word2vec_pickle, 'wb') as f_pickle:
            pickle.dump(word2vec, f_pickle)

    # 从word文档里抽取关键词
    @staticmethod
    def extract_keywords():
        f_raw = docx.Document(FilePool.raw_docx2)
        for para in f_raw.paragraphs:
            if len(para.runs) != 0 and para.runs[-1].bold:
                keyword = para.text.strip()
                print(keyword)

    # 分词
    @staticmethod
    def cut_text(text_list):
        cut_text = []
        for line in text_list:
            cut_line = [w for w in jieba.cut(line)]  # 分词
            cut_text.append(cut_line)
        return cut_text

    # 清洗text(输入为list of str)
    @staticmethod
    def clean_text(text):
        cleaned = []
        for line in text:
            line = Utils.clean_line(line)
            cleaned.append(line)
        return cleaned

    # 清洗text(输入为list of str)
    @staticmethod
    def clean_line(line):
        line = line.lstrip()  # 去掉行首空格
        line = line.rstrip()  # 去掉行尾换行符
        line = line.replace('\t', '')  # 去掉\t
        line = Utils.full2half(line)  # 全角转半角
        line = Utils.str2cn(line)  # 阿拉伯数字转中文
        ret = line.replace(' ', '')
        return ret

    '''以下均为私有方法'''

    @staticmethod
    def __int2cn(matched):
        if len(matched.group(0)) >= 8:
            # 长度 >= 8的阿拉伯数字不转换
            return matched.group(0)
        else:
            return an2cn(matched.group(0), 'low')


# for test purpose
if __name__ == '__main__':
    # Utils.delete_titles('data/long_answers.txt', 'data/tmp.txt')
    # print(Utils.aver_len('data/input.txt'))
    # print(Utils.aver_len('data/cleaned_answers.txt'))
    # print(Utils.aver_len('data/long_answers.txt'))
    Utils.extract_keywords()

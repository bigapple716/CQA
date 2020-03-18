# -*- coding: utf-8 -*-

import os
import platform


# 文件名集合
class FilePool:
    relative_path = 'textqa/model/CQA/'
    project_dir = os.path.join(os.getcwd(), relative_path)

    # 原始文档
    raw_docx1 = os.path.join(project_dir, 'data/长沙机场知识库(没目录图片表格).docx')  # 长沙机场知识文档
    raw_docx2 = os.path.join(project_dir, 'data/96566机场问询资料.docx')  # 昆明机场知识文档
    answers_txt = os.path.join(project_dir, 'data/answers.txt')  # 全部机场知识文档的txt格式集合(不含补充知识文档)
    extra_txt = os.path.join(project_dir, 'data/extra_answers.txt')  # 补充知识文档

    # 清洗过的答案库
    small_answers_txt = os.path.join(project_dir, 'data/small_answers.txt')  # 全部答案 - 长沙机场(没分词)
    cleaned_answers_txt = os.path.join(project_dir, 'data/cleaned_answers.txt')  # 短答案(没分词)
    cleaned_answers_json = os.path.join(project_dir, 'data/cleaned_answers.json')  # 短答案(已分词)
    cleaned_extra_txt = os.path.join(project_dir, 'data/cleaned_extra.txt')  # 补充答案(没分词)
    cleaned_extra_json = os.path.join(project_dir, 'data/cleaned_extra.json')  # 补充答案(已分词)
    long_answers_txt = os.path.join(project_dir, 'data/long_answers.txt')  # 长答案(没分词)
    long_answers_json = os.path.join(project_dir, 'data/long_answers.json')  # 长答案(已分词)

    base_question_file = os.path.join(project_dir, 'data/base_questions.json')

    # 输入输出
    input_txt = os.path.join(project_dir, 'data/input.txt')  # 问题queries，仅在批量测试时用
    output_csv = os.path.join(project_dir, 'data/output.csv')  # 输出的回答，仅在批量测试时用

    stopword_txt = os.path.join(project_dir, 'data/stopword.txt')  # 停用词表

    docx_list = [raw_docx1, raw_docx2]

    word2vec_pickle = os.path.join(project_dir, 'data/word2vec.pickle')

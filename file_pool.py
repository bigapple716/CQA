# -*- coding: utf-8 -*-

import os


# 文件名集合
class FilePool:
    # relative_path = 'textqa/model/CQA/'
    relative_path = ''
    project_dir = os.path.join(os.getcwd(), relative_path)

    raw_docx1 = os.path.join(project_dir, 'data/长沙机场知识库(没目录图片表格).docx')
    raw_docx2 = os.path.join(project_dir, 'data/96566机场问询资料.docx')
    answers_txt = os.path.join(project_dir, 'data/answers.txt')
    extra_txt = os.path.join(project_dir, 'data/extra_answers.txt')
    cleaned_answers_json = os.path.join(project_dir, 'data/cleaned_answers.json')
    cleaned_answers_txt = os.path.join(project_dir, 'data/cleaned_answers.txt')
    cleaned_extra_txt = os.path.join(project_dir, 'data/cleaned_extra.txt')
    cleaned_extra_json = os.path.join(project_dir, 'data/cleaned_extra.json')
    long_answers_txt = os.path.join(project_dir, 'data/long_answers.txt')
    long_answers_json = os.path.join(project_dir, 'data/long_answers.json')
    input_txt = os.path.join(project_dir, 'data/input.txt')
    output_csv = os.path.join(project_dir, 'data/output.csv')
    stopword_txt = os.path.join(project_dir, 'data/stopword.txt')

    docx_list = [raw_docx1, raw_docx2]

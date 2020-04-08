# -*- coding: utf-8 -*-

import os


# 文件名集合
class FilePool:
    relative_path = 'textqa/model/CQA/'
    project_dir = os.path.join(os.getcwd(), relative_path)

    # 原始文档
    raw_docx1 = os.path.join(project_dir, 'data/raw_data/长沙机场知识库(没目录图片表格).docx')  # 长沙机场知识文档
    raw_docx2 = os.path.join(project_dir, 'data/raw_data/96566机场问询资料.docx')  # 昆明机场知识文档
    answers_txt = os.path.join(project_dir, 'data/answers.txt')  # 全部机场知识文档的txt格式集合(不含补充知识文档)
    extra_txt = os.path.join(project_dir, 'data/raw_data/extra_answers.txt')  # 补充知识文档
    raw_small_answers_txt = os.path.join(project_dir, 'data/raw_data/raw_small_answers.txt')  # 全部答案 - 长沙机场

    # 清洗过的答案库
    small_answers_txt = os.path.join(project_dir, 'data/small_answers.txt')  # 全部答案 - 长沙机场(没分词)
    small_answers_json = os.path.join(project_dir, 'data/small_answers.json')  # 全部答案 - 长沙机场(已分词)
    cleaned_answers_txt = os.path.join(project_dir, 'data/cleaned_answers.txt')  # 短答案(没分词)
    cleaned_answers_json = os.path.join(project_dir, 'data/cleaned_answers.json')  # 短答案(已分词)
    cleaned_extra_txt = os.path.join(project_dir, 'data/cleaned_extra.txt')  # 补充答案(没分词)
    cleaned_extra_json = os.path.join(project_dir, 'data/cleaned_extra.json')  # 补充答案(已分词)
    long_answers_txt = os.path.join(project_dir, 'data/long_answers.txt')  # 长答案(没分词)
    long_answers_json = os.path.join(project_dir, 'data/long_answers.json')  # 长答案(已分词)

    # QA库
    qa_question_file = os.path.join(project_dir, 'data/match_question.txt')  # QA对里的原始问题集合
    # QA对里的原始答案集合
    qa_gold_list = [
        os.path.join(project_dir, 'data/match_gold.txt')
    ]
    qa_file = os.path.join(project_dir, 'data/qa.json')  # QA对
    base_ques_list_file = os.path.join(project_dir, 'data/base_ques_list.json')  # QA对里的问题集合

    # 关键词库
    keyword_database_json = os.path.join(project_dir, 'data/keyword_database.json')

    # 输入
    input_txt = os.path.join(project_dir, 'data/input.txt')  # 问题queries，仅在批量测试时用

    # 输出
    output_csv = os.path.join(project_dir, 'data/output.csv')  # 输出的回答，仅在批量测试时用
    output_question_csv = os.path.join(project_dir, 'data/output_questions.csv')  # 被匹配上的问题，仅在QQ匹配时用

    stopword_txt = os.path.join(project_dir, 'data/stopword.txt')  # 停用词表
    user_dict = os.path.join(project_dir, 'data/air_lexicon.txt')  # 分词用的字典

    docx_list = [raw_docx1, raw_docx2]
    keyword_list = [
        os.path.join(project_dir, 'data/keyword/乘机证件.txt'),
        os.path.join(project_dir, 'data/keyword/中转服务.txt'),
        os.path.join(project_dir, 'data/keyword/值机.txt'),
        os.path.join(project_dir, 'data/keyword/国内到达.txt'),
        os.path.join(project_dir, 'data/keyword/物品携带.txt'),
        os.path.join(project_dir, 'data/keyword/安全检查.txt'),
        os.path.join(project_dir, 'data/keyword/机场服务.txt'),
        os.path.join(project_dir, 'data/keyword/物品遗失.txt'),
        os.path.join(project_dir, 'data/keyword/特殊旅客.txt'),
        os.path.join(project_dir, 'data/keyword/行李携带.txt'),
        os.path.join(project_dir, 'data/keyword/其他.txt'),  # 这个文件里只有意图识别的数据
    ]

    # 意图识别QA对
    intention_qa_list = [
        os.path.join(project_dir, 'data/intention/中转服务.txt'),
        os.path.join(project_dir, 'data/intention/值机.txt'),
        os.path.join(project_dir, 'data/intention/其他.txt'),
        os.path.join(project_dir, 'data/intention/安全检查.txt'),
        os.path.join(project_dir, 'data/intention/有效证件.txt'),
        os.path.join(project_dir, 'data/intention/物品遗失.txt'),
        os.path.join(project_dir, 'data/intention/特殊旅客.txt'),
        os.path.join(project_dir, 'data/intention/行李携带.txt'),
    ]

    # 预训练中文词向量
    word2vec_pickle = os.path.join(project_dir, 'data/word2vec.pickle')  # text格式 + pickle格式
    word2vec_bin = os.path.join(project_dir, 'data/word2vec_bin.vector')  # 二进制格式

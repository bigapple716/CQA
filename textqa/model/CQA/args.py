# -*- coding: utf-8 -*-

from textqa.model.CQA.method import Method

# 系统参数集合
# 所有参数应该仅在该文件中可以修改，在其他文件中应为只读状态
method = Method.mix  # 检索方法
trim_stop = True  # 是否去停用词
answer_base = 'small'  # 答案库来源: long, cleaned, small
categorize_question = True  # 是否对问题进行分类
uni_idf = False  # 是否用全集答案库计算IDF
top_n = 3  # 返回前top_n个结果
qq_threshold = 0.7  # QQ匹配的阈值
qa_threshold = 0  # QA匹配的阈值
cat_threshold = 0.9  # 问题分类时new bm25的阈值
cat_adv_norm_threshold = 0.3  # 问题分类 且 用advanced normalization时 new bm25的阈值
syn_threshold = 0.78  # 近义词匹配的阈值
kw_from_graph = False  # 从知识图谱读取分类答案(而不是从本地文件读取)
advanced_norm = False
bm25_new_cat_threshold = 2  # new bm25下问题分类得分的阈值（低于它就不用分类答案库了）

enable_log = False  # 是否向控制台打印log

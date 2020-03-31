# -*- coding: utf-8 -*-

from textqa.model.CQA.method import Method

# 所有系统参数
# 以下的值均为默认值
method = Method.mix  # 检索方法
trim_stop = True  # 是否去停用词
answer_base = 'small'  # 答案库来源: long, cleaned, small
categorize_question = False  # 是否对问题进行分类
top_n = 3  # 返回前top_n个结果
qq_threshold = 0.7  # QQ匹配的阈值
qa_threshold = 5  # QA匹配的阈值
syn_threshold = 0.7  # 近义词匹配的阈值

enable_log = False  # 是否向控制台打印log

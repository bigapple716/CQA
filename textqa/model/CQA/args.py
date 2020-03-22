# -*- coding: utf-8 -*-

# 所有系统参数
# 以下的值均为默认值
method = 'bm25'  # 检索方法: bm25, qq-match, mix, tfidf-sim, aver-embed, lm
trim_stop = True  # 是否去停用词
answer_base = 'small'  # 答案库来源: long, cleaned, small
categorize_question = True  # 是否对问题进行分类
top_n = 3  # 返回前top_n个结果
qq_threshold = 0.7  # QQ匹配的阈值
qa_threshold = 5  # QA匹配的阈值

enable_log = False  # 是否向控制台打印log

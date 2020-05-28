# -*- coding: utf-8 -*-

from textqa.model.CQA.search_algorithms import *
from textqa.model.CQA.reader import Reader
from textqa.model.CQA.post_processor import PostProcessor
from textqa.model.CQA.scripts.evaluate import Evaluate
from textqa.model.CQA.method import Method
from textqa.model.CQA import args
import platform
from absl import logging
import jieba
import warnings


class TBQA:
    def __init__(self):
        if platform.system() == 'Darwin':
            args.enable_log = True

        # 载入用户字典(分词用)
        # jieba.load_userdict(FilePool.user_dict)

        self.reader = Reader()  # 实例化一个Reader类
        # self.reader.preprocess()  # 这行代码只需要运行一次

        self.baseline_model = Baselines()  # 放在这里就只需要初始化一次，能提高性能

        self.post_processor = PostProcessor()  # 实例化一个后处理类

        # 反馈参数设置情况
        if args.enable_log:
            print('==========> TBQA settings <==========')
            print('method:', args.method)
            print('trim stop words:', args.trim_stop)
            print('answer base:', args.answer_base)
            print('categorize questions:', args.categorize_question)
            print('universal IDF:', args.uni_idf)
            print('return top ' + str(args.top_n) + ' results')
            print('QQ threshold:', args.qq_threshold)
            print('QA threshold:', args.qa_threshold)
            print('cat threshold:', args.cat_threshold)
            print('cat with advanced norm threshold:', args.cat_adv_norm_threshold)
            print('syn threshold:', args.syn_threshold)
            print('read from graph:', args.kw_from_graph)
            print('new BM25 uses advanced normalization:', args.advanced_norm)
            print('logging enabled')
            print('=====================================')

    def search_answers(self, cleaned_in, uncut_in):
        sorted_scores_list = []
        answers_list = []
        answers_index_list = []
        questions_list = []
    
        for i, (cut_query, uncut_query) in enumerate(zip(cleaned_in, uncut_in)):
            # 答案来源
            if args.categorize_question:
                categorized_qa = self.reader.categorize(uncut_query)
            else:
                categorized_qa = None

            # 用不同算法搜索
            if args.method == Method.bm25:
                sorted_scores, max_pos, answers = self.baseline_model.bm25(cut_query, categorized_qa)
            elif args.method == Method.bm25_syn:
                sorted_scores, max_pos, answers = self.baseline_model.bm25_syn(cut_query)
            elif args.method == Method.bm25_new:
                sorted_scores, max_pos, answers = self.baseline_model.bm25_new(cut_query, uncut_query, categorized_qa)
            elif args.method == Method.qq_match:
                sorted_scores, max_pos, answers, questions = self.baseline_model.qq_match(cut_query)
                questions_list.append(questions)
            elif args.method == Method.mix:
                sorted_scores, max_pos, answers, questions = \
                    self.baseline_model.qq_cat_qa_filter(cut_query, uncut_query, categorized_qa)
                questions_list.append(questions)
            elif args.method == Method.tfidf_sim:
                sorted_scores, max_pos, answers = self.baseline_model.tfidf_sim(cut_query)
            else:
                raise Exception('尚未支持该搜索算法！')
    
            sorted_scores_list.append(sorted_scores[:args.top_n])
            answers_list.append(answers[:args.top_n])
            answers_index_list.append(max_pos[:args.top_n])

            # 输出实时进度
            if args.enable_log and i % 100 == 0:
                print('line ' + str(i) + ' processed')

        if args.method == Method.qq_match or args.method == Method.mix:
            self.post_processor.print_answers(questions_list, FilePool.output_question_csv)

        return answers_list, answers_index_list, sorted_scores_list

    # 对外接口，输入1个问题给出1个回答
    def get_answer(self, question):
        # 清洗输入
        cleaned_input, uncut_input = self.reader.clean_cut_trim([question])

        # 检索答案
        answers_list, answer_idx_list, sorted_scores_list = self.search_answers(cleaned_input, uncut_input)

        # 清洗答案
        # if args.method != 'qq-match' and args.method != 'mix':
        #     answers_list = self.post_processor.clean_answers(list(answers_list), list(answer_idx_list))  # 清洗答案

        answers = answers_list[0]
        scores = sorted_scores_list[0]

        if args.enable_log:
            print(scores[:args.top_n])

        # 如果回答列表是空的，那么说明这个问题不可回答
        if len(answers) == 0:
            answers.append(-1)

        return answers

    # 测试用接口，批量回答问题
    def get_multi_answers(self):
        self.__clear_output_file()

        # 读入输入问题
        with open(FilePool.input_txt, 'r') as f_input:
            doc = f_input.readlines()
        questions = [line.rstrip('\n') for line in doc]

        # 清洗输入
        cleaned_input, uncut_input = self.reader.clean_cut_trim(questions)

        # 检索答案
        answers_list, answer_idx_list, sorted_scores_list = self.search_answers(cleaned_input, uncut_input)

        # 清洗答案
        # if args.method != 'qq-match' and args.method != 'mix':
        #     answers_list = self.post_processor.clean_answers(list(answers_list), list(answer_idx_list))  # 清洗答案

        empty_ans = 0
        for scores, ans in zip(sorted_scores_list, answers_list):
            if args.enable_log:
                print(scores[:args.top_n])
            # 如果回答列表是空的，那么说明这个问题不可回答
            if len(ans) == 0:
                empty_ans += 1
                ans.append(-1)

        print('reject rate:', empty_ans / len(sorted_scores_list))

        self.post_processor.print_answers(answers_list, FilePool.output_csv)

    # 清空输出文件
    def __clear_output_file(self):
        with open(FilePool.output_csv, 'w') as f_output:
            pass
        with open(FilePool.output_question_csv, 'w') as f:
            pass


if __name__ == '__main__':
    # 设置日志级别
    logging.set_verbosity('error')
    warnings.filterwarnings('ignore')

    tbqa = TBQA()

    # question = '身份证丢了怎么登机'
    # answer = tbqa.get_answer(question)
    # print('answer:', answer)

    tbqa.get_multi_answers()

    # 自动评测
    evaluator = Evaluate()
    evaluator.evaluate()

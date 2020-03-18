# -*- coding: utf-8 -*-

from textqa.model.CQA.search_algorithms import *
from textqa.model.CQA.reader import Reader
from textqa.model.CQA.post_processor import PostProcessor
from textqa.model.CQA import args


class TBQA:
    def __init__(self, method='mix', trim_stop=True, long_ans=True, top_n=3, qq_threshold=0.6, qa_threshold=5.0):
        args.method = method
        args.trim_stop = trim_stop
        args.long_ans = long_ans
        args.top_n = top_n

        self.qa_threshold = qa_threshold
        self.qq_threshold = qq_threshold

        self.reader = Reader(args.trim_stop)  # 实例化一个Reader类
        self.reader.preprocess()

        self.post_processor = PostProcessor()  # 实例化一个后处理类

        # 反馈参数设置情况
        print('==========> TBQA settings <==========')
        print('method:', args.method)
        print('trim stop words:', args.trim_stop)
        print('use long answers:', args.long_ans)
        print('return top ' + str(args.top_n) + 'results')
        print('QQ threshold:', self.qq_threshold)
        print('QA threshold:', self.qa_threshold)

    def search_answers(self, cleaned_in, uncut_in,):
        baseline_model = Baselines()
    
        sorted_scores_list = []
        answers_list = []
        answers_index_list = []
        questions_list = []
    
        for i, (cut_query, query) in enumerate(zip(cleaned_in, uncut_in)):
            # 用不同算法搜索
            if args.method == 'bm25':
                sorted_scores, max_pos, answers = baseline_model.bm25(cut_query, baseline_model.cut_answers)
            elif args.method == 'qq-match':
                sorted_scores, max_pos, answers, questions = baseline_model.qq_match(cut_query)
                questions_list.append(questions)
            elif args.method == 'mix':
                sorted_scores, max_pos, answers, questions = baseline_model.qq_qa_mix(cut_query)
                questions_list.append(questions)
            elif args.method == 'tfidf-sim':
                sorted_scores, max_pos, answers = baseline_model.tfidf_sim(cut_query, baseline_model.cut_answers)
            # elif args.method == 'tfidf-dist':
            #     result = Baselines.tfidf_dist(cut_query, cleaned_ans_json)
            # elif args.method == 'tfidf':
            #     result = Baselines.tfidf(cut_query, cleaned_ans_txt)
            elif args.method == 'aver-embed':
                sorted_scores, max_pos, answers = baseline_model.aver_embed(cut_query)
            elif args.method == 'lm':
                sorted_scores, max_pos, answers = baseline_model.language_model(cut_query)
            else:
                raise Exception('尚未支持该搜索算法！')
    
            sorted_scores_list.append(sorted_scores)
            answers_list.append(answers)
            answers_index_list.append(max_pos)
        '''
            # 输出实时进度
            if i % 20 == 0:
                print('line ' + str(i) + ' processed')
        
        if method == 'qq-match' or 'mix':
            print_answers(questions_list, 'data/output_questions.csv')
        '''
        return answers_list, answers_index_list, sorted_scores_list

    # 对外接口，输入1个问题给出1个回答
    def get_answer(self, question):
        cleaned_input, uncut_input = self.reader.clean_input([question])

        if args.long_ans:
            # using long answers
            answers_list, answer_idx_list, sorted_scores_list = \
                self.search_answers(cleaned_input, uncut_input)
            if args.method != 'qq-match' and args.method != 'mix':
                answers_list = self.post_processor.clean_answers(list(answers_list), list(answer_idx_list), FilePool.long_answers_txt)  # 清洗答案
        else:
            # NOT using long answers
            answers_list, answer_idx_list, sorted_scores_list = \
                self.search_answers(cleaned_input, uncut_input)
            if args.method != 'qq-match' and args.method != 'mix':
                answers_list = self.post_processor.clean_answers(list(answers_list), list(answer_idx_list), FilePool.cleaned_answers_txt)  # 清洗答案

        answers = answers_list[0]
        scores = sorted_scores_list[0]

        print(scores[:args.top_n])

        if len(answers) == 0:
            answers.append(-1)
        return answers

    # 测试用接口，批量回答问题
    def get_multi_answers(self):
        pass


if __name__ == '__main__':
    tbqa = TBQA()
    question = '接机攻略'
    
    print('question:', question)
    answer = tbqa.get_answer(question)
    print('answer:', answer)

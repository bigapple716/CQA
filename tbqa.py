import csv
import re
import argparse

from search_algorithms import *
from reader import Reader


class TBQA:
    def __init__(self, method='mix', trim_stop=True, long_ans=True, top_n=3, qq_threshold=0.6, qa_threshold=5.0):
        self.method = method
        self.trim_stop = trim_stop
        self.long_ans = long_ans
        self.top_n = top_n
        self.qa_threshold = qa_threshold
        self.qq_threshold = qq_threshold

        raw_docx1 = 'data/长沙机场知识库(没目录图片表格).docx'
        raw_docx2 = 'data/96566机场问询资料.docx'
        answers_txt = 'data/answers.txt'
        extra_txt = 'data/extra_answers.txt'
        cleaned_answers_json = 'data/cleaned_answers.json'
        cleaned_answers_txt = 'data/cleaned_answers.txt'
        cleaned_extra_txt = 'data/cleaned_extra.txt'
        cleaned_extra_json = 'data/cleaned_extra.json'
        long_answers_txt = 'data/long_answers.txt'
        long_answers_json = 'data/long_answers.json'
        input_txt = 'data/input.txt'
        output_csv = 'data/output.csv'
        stopword_txt = 'data/stopword.txt'

        self.reader = Reader(self.trim_stop, stopword_txt, input_txt, [raw_docx1, raw_docx2], answers_txt, extra_txt,
                             cleaned_answers_txt, cleaned_answers_json,
                             long_answers_txt, long_answers_json,
                             cleaned_extra_txt, cleaned_extra_json)  # 实例化一个Reader类
        #print('init reader')
        #self.reader.preprocess()
        #print('finish reader')

    def search_answers(self,cleaned_in, uncut_in, cleaned_ans_json, cleaned_ans_txt):
        baseline_model = Baselines(cleaned_ans_json, cleaned_ans_txt)
    
        sorted_scores_list = []
        answers_list = []
        answers_index_list = []
        questions_list = []
    
        for i, (cut_query, query) in enumerate(zip(cleaned_in, uncut_in)):
            # 用不同算法搜索
            if self.method == 'bm25':
                sorted_scores, max_pos, answers = baseline_model.bm25(cut_query, baseline_model.cut_answers)
            elif self.method == 'qq-match':
                sorted_scores, max_pos, answers, questions = baseline_model.qq_match(cut_query)
                questions_list.append(questions)
            elif self.method == 'mix':
                sorted_scores, max_pos, answers, questions = baseline_model.qq_qa_mix(cut_query)
                questions_list.append(questions)
            elif self.method == 'tfidf-sim':
                sorted_scores, max_pos, answers = baseline_model.tfidf_sim(cut_query, baseline_model.cut_answers)
            # elif method == 'tfidf-dist':
            #     result = Baselines.tfidf_dist(cut_query, cleaned_ans_json)
            # elif method == 'tfidf':
            #     result = Baselines.tfidf(cut_query, cleaned_ans_txt)
            elif self.method == 'aver-embed':
                sorted_scores, max_pos, answers = baseline_model.aver_embed(cut_query)
            elif self.method == 'lm':
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
        
        if method == 'mix':
            print('QQ count:', baseline_model.qq_count)
            print('QA count:', baseline_model.qa_count)
        if method == 'qq-match' or 'mix':
            print_answers(questions_list, 'data/output_questions.csv')
        '''
        return answers_list, answers_index_list, sorted_scores_list

    def clean_answers(self,ans_in, ans_idx_in, cleaned_ans_txt):
        # 入参检查
        if len(ans_in) != len(ans_idx_in):
            raise Exception('答案列表和答案索引列表长度不一致！')
    
        # 载入答案文档
        with open(cleaned_ans_txt, 'r') as f_ans_txt:
            text = f_ans_txt.readlines()
    
        ret = []
        for answers, answer_indexes in zip(ans_in, ans_idx_in):
            cleaned_ans = []
            for ans, ans_idx in zip(answers, answer_indexes):
                if re.match(r'\d+\.\d+', ans) is not None:
                    # 如果答案是二级标题，那么就取下一个段落(下个段落有可能是三级标题)
                    next_ans = text[ans_idx + 1].rstrip()
                    next_idx = ans_idx + 1
                    # print('标题：' + ans + ' -> 内容：' + next_ans)
                    # 再判断一下是不是三级标题
                    if re.match(r'\d+\.\d+\.\d+', next_ans) is not None:
                        real_ans = text[next_idx + 1].rstrip()
                        cleaned_ans.append(real_ans)
                        # print('标题：' + next_ans + ' -> 内容：' + real_ans)
                    else:
                        cleaned_ans.append(next_ans)
    
                elif re.match(r'\d+\.\d+\.\d+', ans) is not None:
                    # 如果答案是三级标题，那么下一个段落就是真正的答案
                    real_ans = text[ans_idx + 1].rstrip()
                    cleaned_ans.append(real_ans)
                    # print('标题：' + ans + ' -> 内容：' + real_ans)
    
                elif re.match('.+([？?])$', ans) is not None:
                    # 如果答案以问号结尾，那么就跳过
                    continue
    
                else:
                    # 如果以上条件都不符合，那么这个答案正常保留
                    cleaned_ans.append(ans)
    
            ret.append(cleaned_ans)
        return ret
    def get_answer(self,question):
        """

        Returns
        -------
        res:返回问题的topn答案，默认为top3
        """
        cleaned_input, uncut_input = self.reader.clean_input([question])
        if self.long_ans:
            #print('using long answers')
            answers_list, answer_idx_list, sorted_scores_list = \
                self.search_answers(cleaned_input, uncut_input,self.reader.long_answers_json,self.reader.long_answers_txt)
            if self.method != 'qq-match' and self.method != 'mix':
                answers_list = clean_answers(list(answers_list), list(answer_idx_list), self.reader.long_answers_txt)  # 清洗答案
        else:
            #print('NOT using long answers')
            answers_list, answer_idx_list, sorted_scores_list = \
                self.search_answers(cleaned_input, uncut_input, cleaned_answers_json, cleaned_answers_txt)
            if method != 'qq-match' and method != 'mix':
                answers_list = self.clean_answers(list(answers_list), list(answer_idx_list), cleaned_answers_txt)  # 清洗答案
        answers = answers_list[0]
        scores = sorted_scores_list[0]
        print(scores[:3])
        if len(answers) == 0:
            answers.append(-1)
        return answers


if __name__ == '__main__':
    tbqa = TBQA()
    questions = ['接机攻略']
    

    #question='长沙黄花机场可以直接买飞机票吗'
    for question in questions:
        print('question',question)
        answers = tbqa.get_answer(question)
        for ele in answers:
            print(ele)

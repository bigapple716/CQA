# -*- coding: utf-8 -*-

import random
import json
from textqa.model.CQA.file_pool import FilePool

# set random seed
random.seed(0)


class DataMaker:
    # file names
    question_file = 'textqa/model/CQA/data/question.txt'
    gold_file = 'textqa/model/CQA/data/gold.txt'
    # answer_file = 'textqa/model/CQA/data/cleaned_answers.txt'
    answer_file = 'textqa/model/CQA/data/long_answers.txt'
    queries_file = 'textqa/model/CQA/data/queries.txt'

    match_question_file = 'textqa/model/CQA/data/match_question.txt'
    match_gold_file = 'textqa/model/CQA/data/match_gold.txt'
    input_txt = 'textqa/model/CQA/data/input.txt'

    # make qa data
    def make_qa_data(self, query_file=FilePool.input_txt):
        # 读文件
        with open(query_file, 'r') as f_query:
            queries = f_query.readlines()
            queries = [line.rstrip('\n') for line in queries]
        with open(self.match_question_file, 'r') as f_ques:
            ques_list = f_ques.readlines()
            ques_list = [line.rstrip('\n') for line in ques_list]
        with open(self.match_gold_file, 'r') as f_gold:
            gold_list = f_gold.readlines()
            gold_list = [line.rstrip('\n') for line in gold_list]

        # 判断上面3个list长度一致
        if not len(ques_list) == len(gold_list):
            raise Exception('数据长度不一致！')

        # 制作字典
        base_questions = []
        for ques, gold in zip(ques_list, gold_list):
            if '五一广场' in gold or '黄花' in gold or '小吃' in gold or '宜居' in gold or '长沙' in gold:
                # print(gold)
                continue
            # 只要问题不在input里面而且带答案的
            if (ques not in queries) and (gold != ''):
                base_questions.append({'question': ques, 'sentence': [gold]})

        # 写到json里
        with open(FilePool.qa_file, 'w') as f_out:
            json.dump(base_questions, f_out, ensure_ascii=False)

    def qa_match(self):
        questions, golds, answers = self.__read(self.question_file, self.gold_file, self.answer_file,
                                                read_gold=True, read_answer=True)
        print('answer pool length:', len(answers))
        train, dev, test, test_questions = self.__create_qa_data(questions, golds, answers,
                                                                 mode='mix', train_dev_ratio=0.9)

        # 写数据到.tsv文件
        self.__write_data(train, 'train.tsv')
        self.__write_data(dev, 'dev.tsv')
        self.__write_data(test, 'test.tsv')
        self.__write_questions(test_questions, 'test_questions.txt')

    def qq_match(self, ratio=0.7):
        questions, golds, _ = self.__read(self.question_file, self.gold_file, self.queries_file,
                                          read_gold=True, read_answer=False)
        answerable = self.__pick_answerable(questions, golds)

        random.shuffle(answerable)  # shuffle

        # 划分哪些问题用来做question哪些用来做query
        size = round(len(answerable) * ratio)
        base_questions = answerable[:size]
        queries = answerable[size:]

        test_data = []
        for query in queries:
            for dict in base_questions:
                data = {'question': query['question'], 'sentence': dict['question'], 'label': 'entailment'}
                test_data.append(data)

        self.__write_data(test_data, 'test.tsv')
        self.__write_questions(queries, 'queries.tsv')
        self.__write_questions(base_questions, 'base_questions.tsv')
        with open('queries.json', 'w') as f_queries:
            json.dump(queries, f_queries, ensure_ascii=False)
        with open('base_questions.json', 'w') as f_base_ques:
            json.dump(base_questions, f_base_ques, ensure_ascii=False)

    def __read(self, question_file, gold_file, answer_file, read_gold=True, read_answer=True):
        # read questions
        with open(question_file, 'r') as f_q:
            questions = f_q.readlines()

        if read_gold:
            # 要读的数据是训练用的带标签数据
            with open(gold_file, 'r') as f_gold:
                golds = f_gold.readlines()
        else:
            # 要读的数据是预测用的无标签数据
            golds = []

        # read the answer pool
        if read_answer:
            with open(answer_file, 'r') as f_ans:
                answers = f_ans.readlines()
        else:
            answers = []

        # strip '\n' and '\t'
        questions = [line.rstrip('\n').replace('\t', '    ').replace('"', '') for line in questions]
        golds = [line.rstrip('\n').replace('\t', '    ').replace('"', '') for line in golds]
        answers = [line.rstrip('\n').replace('\t', '    ').replace('"', '') for line in answers]

        # check if questions and gold have the same length
        if read_gold and len(questions) != len(golds):
            raise Exception('问题数量和正确答案数量不相等！')

        return questions, golds, answers

    def __create_qa_data(self, questions, golds, answers, mode='train', train_dev_ratio=0.7, neg_pos_ratio=1):
        """

        Parameters
        ----------
        questions
        golds
        answers
        train_dev_ratio : float
            train数据和dev数据的比例
        mode : str
            生成数据的模式：train, test, mix
        neg_pos_ratio : int
            负样例 / 正样例

        Returns
        -------

        """
        # 4个返回值
        train_data = []
        dev_data = []
        test_data = []
        dev_answerable = []

        if mode == 'train' or mode == 'mix':
            # 筛选出可回答的问题
            answerable_list = self.__pick_answerable(questions, golds)

            # shuffle
            random.shuffle(answerable_list)

            # 计算训练集大小
            train_size = round(len(answerable_list) * train_dev_ratio)
            # 划分训练集和验证集
            train_answerable = answerable_list[:train_size]
            dev_answerable = answerable_list[train_size:]

            for dict in train_answerable:
                # dict本身就是一条positive的数据
                train_data.append(dict)
                # 创建若干条negative的数据
                for i in range(neg_pos_ratio):
                    neg_ans = answers[random.randint(0, len(answers) - 1)]
                    dict_neg = {'question': dict['question'], 'sentence': neg_ans, 'label': 'not_entailment'}
                    train_data.append(dict_neg)

            if mode == 'train':
                for dict in dev_answerable:
                    # dict本身就是一条positive的数据
                    dev_data.append(dict)
                    # 创建若干条negative的数据
                    for i in range(neg_pos_ratio):
                        neg_ans = answers[random.randint(0, len(answers) - 1)]
                        dict_neg = {'question': dict['question'], 'sentence': neg_ans, 'label': 'not_entailment'}
                        dev_data.append(dict_neg)
            else:
                # mode == 'mix'
                for dict in dev_answerable:
                    for ans in answers:
                        # 其实这里的label不重要，反正在test集里也没人看label
                        test_dict = {'question': dict['question'], 'sentence': ans, 'label': 'entailment'}
                        test_data.append(test_dict)

        else:
            # 标签列表golds是空，说明是不带标签的待预测的数据
            for ques in questions:
                for ans in answers:
                    dict = {'question': ques, 'sentence': ans, 'label': 'entailment'}
                    test_data.append(dict)

        random.shuffle(train_data)
        random.shuffle(dev_data)

        return train_data, dev_data, test_data, dev_answerable

    # 筛选出可回答的问题
    def __pick_answerable(self, questions, golds):
        answerable_list = []
        for ques, gold in zip(questions, golds):
            if gold != '':
                dict = {'question': ques, 'sentence': gold, 'label': 'entailment'}
                answerable_list.append(dict)
        return answerable_list

    def __write_data(self, dict_list, filename):
        with open(filename, 'w') as f_data:
            # 1st line of the file
            f_data.write('index\tquestion\tsentence\tlabel\n')

            for idx, dict in enumerate(dict_list):
                f_data.write(str(idx) + '\t' + dict['question'] + '\t' + dict['sentence'] + '\t' + dict['label'])
                f_data.write('\n')

    def __write_questions(self, dict_list, filename):
        with open(filename, 'w') as f_ques:
            for dict in dict_list:
                f_ques.write(dict['question'])
                f_ques.write('\n')


if __name__ == "__main__":
    data_maker = DataMaker()

    data_maker.make_qa_data()
    # pre_processor.qq_match()
    # pre_processor.qa_match()

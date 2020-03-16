# -*- coding: utf-8 -*-

import random
import json

# set random seed
random.seed(0)


class PreProcessor:
    # file names
    question_file = 'data/question.txt'
    gold_file = 'data/gold.txt'
    # answer_file = 'data/cleaned_answers.txt'
    answer_file = 'data/long_answers.txt'
    queries_file = 'data/queries.txt'

    match_qid_file = 'data/match_qid.txt'
    match_question_file = 'data/match_question.txt'
    match_gold_file = 'data/match_gold.txt'
    input_txt = 'data/input.txt'

    # 根据问题ID找问答对
    def qid2qa(self, input_file='data/qid.txt', output_file='data/base_questions.json'):
        # constants
        ratio = 0.7

        # 读文件
        with open(self.match_qid_file, 'r') as f_qid:
            qid_list = f_qid.readlines()
            qid_list = [line.rstrip('\n') for line in qid_list]
        with open(self.match_question_file, 'r') as f_ques:
            ques_list = f_ques.readlines()
            ques_list = [line.rstrip('\n') for line in ques_list]
        with open(self.match_gold_file, 'r') as f_gold:
            gold_list = f_gold.readlines()
            gold_list = [line.rstrip('\n') for line in gold_list]

        # 判断上面3个list长度一致
        if not (len(qid_list) == len(ques_list) == len(gold_list)):
            raise Exception('数据长度不一致！')

        # 制作字典
        qa_dict = {}
        for qid, ques, gold in zip(qid_list, ques_list, gold_list):
            qa_dict[qid] = {'question': ques, 'gold': gold}

        # 读待匹配的问题ID
        with open(input_file, 'r') as f_in:
            ids = f_in.readlines()
            ids = [line.rstrip('\n') for line in ids]

        # 匹配
        small_qa_dict = {i: qa_dict[i] for i in ids}

        # 整理格式
        qa_list = []
        for key in small_qa_dict:
            dict = {'question': small_qa_dict[key]['question'], 'sentence': small_qa_dict[key]['gold']}
            qa_list.append(dict)

        # 划分base_questions和queries
        random.shuffle(qa_list)
        split = round(len(qa_list) * ratio)
        base_questions = qa_list[:split]
        queries = qa_list[split:]

        # 写到json里
        with open(output_file, 'w') as f_out:
            json.dump(base_questions, f_out, ensure_ascii=False)
        with open(self.input_txt, 'w') as f_query:
            for dict in queries:
                f_query.write(dict['question'] + '\n')

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

        if mode == 'train' or 'mix':
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
    pre_processor = PreProcessor()

    pre_processor.qid2qa()
    # pre_processor.qq_match()
    # pre_processor.qa_match()

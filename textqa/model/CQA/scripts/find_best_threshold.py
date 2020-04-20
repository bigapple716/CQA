# -*- coding: utf-8 -*-

from textqa.model.CQA.tbqa import TBQA
from textqa.model.CQA.scripts.evaluate import Evaluate
from textqa.model.CQA import args
from absl import logging
import warnings


if __name__ == '__main__':
    # 设置日志级别
    logging.set_verbosity('error')
    warnings.filterwarnings('ignore')

    args.cat_threshold = 0.8
    while args.cat_threshold > 0:
        print('cat_threshold:', args.cat_threshold)
        tbqa = TBQA()
        tbqa.get_multi_answers()
        evaluator = Evaluate()
        evaluator.evaluate()
        args.cat_threshold -= 0.1

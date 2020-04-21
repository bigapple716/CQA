# -*- coding:utf-8 -*-

from textqa.model.CQA.tbqa import TBQA
import time
import json
from absl import logging
import warnings


# 性能测试
def performance_test(text):
    start = time.time()
    tbqa = TBQA()
    end = time.time()
    print('creating model time:', end - start)
    start = time.time()
    res = tbqa.get_answer(text)
    json_data = json.dumps(res, ensure_ascii=False)
    end = time.time()
    print('get_answer time:', end-start)
    return json_data


# 测试脚本
if __name__ == '__main__':
    # 设置日志级别
    logging.set_verbosity('error')
    warnings.filterwarnings('ignore')

    performance_test('出生证明证件号')

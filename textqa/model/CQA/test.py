# -*- coding:utf-8 -*-

from textqa.model.CQA.tbqa import TBQA
import time
import json


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


performance_test('我可以带生鸡蛋上飞机么')

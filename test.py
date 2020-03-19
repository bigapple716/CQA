# -*- coding:utf-8 -*-
from textqa.model.CQA import tbqa
import time
import json
def textqa(text):
    start = time.time()
    tb = tbqa.TBQA()
    end = time.time()
    print('create time:',end - start)
    start = time.time()
    res = tb.get_answer(text)
    json_data = json.dumps(res, ensure_ascii=False)
    end = time.time()
    print('get_answer time:',end-start)
    return json_data

textqa('我可以带生鸡蛋上飞机么')

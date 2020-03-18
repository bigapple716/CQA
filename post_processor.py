# -*- coding: utf-8 -*-

import re
import csv


class PostProcessor:
    def clean_answers(self, ans_in, ans_idx_in, cleaned_ans_txt):
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

    def print_answers(self, ans_list, output):
        with open(output, 'a') as f_out:
            writer = csv.writer(f_out, lineterminator='\n')
            for answers in ans_list:
                writer.writerow(answers)

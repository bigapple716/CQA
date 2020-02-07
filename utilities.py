# -*- coding: utf-8 -*-


# 统计文件里平均行长度
def aver_len(input_file):
    with open(input_file, 'r') as f_in:
        text = f_in.readlines()

    total_len = 0
    for line in text:
        total_len += len(line.rstrip())
    return total_len / len(text)


print(aver_len('data/input.txt'))
print(aver_len('data/cleaned_answers.txt'))
print(aver_len('data/long_answers.txt'))

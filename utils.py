# -*- coding: utf-8 -*-

from cn2an import an2cn
import re


# 静态工具类
class Utils:
    @staticmethod
    def trim_result(score_list, pos_list, threshold, bigger_is_better=True):
        """
        得分小于/大于threshold的回答直接丢弃

        Parameters
        ----------
        score_list
        pos_list
        threshold
        bigger_is_better : boolean
            得分越高越好/丢弃掉得分过低的回答；默认为True

        Returns
        -------

        """
        if bigger_is_better:
            # 得分 小于 threshold的回答要被丢弃
            for i in range(len(score_list)):
                if score_list[i] < threshold:
                    pos_list[i] = -1
        else:
            # 得分 大于 threshold的回答要被丢弃
            for i in range(len(score_list)):
                if score_list[i] > threshold:
                    pos_list[i] = -1
        return pos_list

    # 判断一个句子是不是标题
    @staticmethod
    def is_heading(sentence):
        if re.match(r'第.{1,2}节', sentence) is not None:
            # 是节标题
            return True
        elif re.match(r'第.{1,2}章', sentence) is not None:
            # 是章标题
            return True
        elif re.match(r'\d+\.\d+', sentence) is not None:
            # 是2级标题
            return True
        elif re.match(r'\d+\.\d+\.\d+', sentence) is not None:
            # 是3级标题
            return True
        else:
            # 肯定不是标题
            return False

    # 从字符串里提取阿拉伯数字并转成中文
    @staticmethod
    def str2cn(str):
        pattern = r'\d+'
        return re.sub(pattern, Utils.__int2cn, str, count=0)

    @staticmethod
    def __int2cn(matched):
        return an2cn(Utils.__full2half(matched.group(0)), 'low')

    # 全角 -> 半角
    @staticmethod
    def __full2half(full_str):
        half_str = ''
        for uchar in full_str:
            code = ord(uchar)
            if code == 12288:  # 全角空格直接转换
                code = 32
            elif 65281 <= code <= 65374:  # 全角字符（除空格）根据关系转化
                code -= 65248
            half_str += chr(code)
        return half_str


# for test purpose
if __name__ == '__main__':
    input1 = '黑黑黑'
    input2 = '囧3私2１sdf45'
    print(Utils.str2cn(input1))
    print(Utils.str2cn(input2))
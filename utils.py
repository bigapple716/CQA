# -*- coding: utf-8 -*-


# 静态工具类
class Utils:
    # 得分小于threshold的回答直接丢弃
    @staticmethod
    def trim_result(score_list, pos_list, threshold):

        return trimed_pos_list
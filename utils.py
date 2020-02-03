# -*- coding: utf-8 -*-


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

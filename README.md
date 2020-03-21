# 张明辉毕业设计全部代码集合

文件列表：
* /data: 通用数据的目录
* /scripts: 脚本所在目录(该目录下的程序单独运行)
    * data_maker.py: 构建不同范围、用途的数据
* reader.py: 读入数据，构建答案集
* utils.py: 一些工具，如：全角转半角，阿拉伯数字转中文数字...
* search_algorithms.py: 答案选择的算法
* pre_processor.py: 数据预处理
* post_processor.py: 答案清洗等后处理工作
* args.py: 参数设置
* file_pool.py: 文件名的集合
* tbqa.py: 主程序，对外接口也在里面
* test.py: 测试代码(单独运行)

运行程序：`python tbqa.py`
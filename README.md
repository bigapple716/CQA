# 张明辉毕业设计全部代码集合 Code of my graduation design

# 文件列表：
* `/data`: 数据目录
    * `/intention`: 意图识别数据
    * `/keyword`: 按照关键词分类的答案
    * `/raw_data`: 原始数据(不是程序生成的文件)
* `/scripts`: 脚本目录(该目录下的程序单独运行)
    * `data_maker.py`: 构建不同范围、用途的数据
    * `evaluate.py`: 自动评测
    * `eval_result.txt`: 评测结果
    * `test.py`: 测试代码
    * `find_best_threshold.py`: 调参用代码
* `args.py`: 参数设置
* `file_pool.py`: 文件名的集合
* `method.py`: 方法枚举类
* `new_bm25.py`: 改进版bm25的模型
* `post_processor.py`: 答案清洗等后处理工作
* `reader.py`: 读入数据和数据预处理
* `search_algorithms.py`: 答案选择的算法
* `text_cnn.py`: Text CNN模型
* `utils.py`: 工具静态类
* `tbqa.py`: 主程序，对外接口也在里面

# 如何运行程序？
`python tbqa.py`

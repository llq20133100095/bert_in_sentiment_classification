# -*- coding: utf-8 -*-
"""
@time: 2019.7.1
@author: liluoqin
@function:
    1.利用结巴提取关键词
"""
from jieba import analyse
import pandas as pd

if __name__ == "__main__":
    # 引入TextRank关键词抽取接口
    textrank = analyse.textrank

    # # 原始文本
    # text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
    #         是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
    #         线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\
    #         线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\
    #         同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"
    #
    # print("\nkeywords by textrank:")
    # # 基于TextRank算法进行关键词抽取
    # keywords = textrank(text, topK=10)
    # # 输出抽取出的关键词
    # for keyword in keywords:
    #     print(keyword + "/")

    file_open = open("./data/train_data.csv", encoding="gbk")
    data = pd.read_csv(file_open)

    neg_content = data[data["label"] == '1']
    neg_content = neg_content["scontent"]
    neg_content = " ".join(neg_content.to_list())

    # 基于TextRank算法进行关键词抽取
    keywords = textrank(neg_content, topK=10)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword + "/")
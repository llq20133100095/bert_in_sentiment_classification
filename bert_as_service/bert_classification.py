# -*- coding: utf-8 -*-
"""
@time: 2019.7.3
@author: liluoqin
@function:
    1.bert embedding in classification
"""
from bert_serving.client import BertClient
import pandas as pd
import sklearn

if __name__ == "__main__":
    # read the all data
    file_open = open("../data/train_data.csv", encoding="gbk")
    data = pd.read_csv(file_open)
    label = data['label']
    text = data['scontent']

    #get embedding
    bc = BertClient()
    feature = bc.encode(text.to_list()[0:2])
    print(feature[0])
    print(feature.shape)
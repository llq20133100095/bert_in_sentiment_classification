# -*- coding: utf-8 -*-
"""
@time: 2019.7.1
@author: liluoqin
@function:
    1.bert embedding in classification
"""
from bert_serving.client import BertClient


if __name__ == "__main__":
    bc = BertClient()
    feature = bc.encode(['First do it', 'then do it right', 'then do it better'])
    print(feature.shape)
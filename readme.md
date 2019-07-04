# bert in sentiment classification

## 0.激活bert环境

```angular2
cd /home/zhangxiang2/.local/bin
字向量:
python3 bert-serving-start -pooling_strategy NONE -model_dir ~/liluoqin/bert_as_service/chinese_L-12_H-768_A-12 -num_worker=4
句向量：
python3 bert-serving-start -model_dir ~/liluoqin/bert_as_service/chinese_L-12_H-768_A-12 -num_worker=4
```

- 其中“chinese_L-12_H-768_A-12”是bert预训练好的模型

## 1.bert_as_service文件夹

### 1.1 bert_as_service/download_glue_data.py

- 下载 GLUE data
- 运行命令：
```angular2
python3 download_glue_data.py --data_dir glue_data --tasks all
```

### 1.2 example_bert_use.py

- 使用bert构建分类器的例子

## 2.代码

### 2.1 jieba_keywords.py

- 用来计算关键词的代码

## 3.数据

- CAIL2018_ALL_DATA
```angular2
wget https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip
```
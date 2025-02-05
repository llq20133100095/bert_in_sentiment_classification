# -*- coding: utf-8 -*-
"""
@time: 2019.7.3
@author: liluoqin
@function:
    1.bert embedding in classification
"""
from bert_serving.client import BertClient
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, auc
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import os
import time
import numpy as np

# from bert_serving.server.helper import get_args_parser
# from bert_serving.server import BertServer
# args = get_args_parser().parse_args(['-model_dir', '/data1/zhangxiang2/liluoqin/bert_as_service/chinese_L-12_H-768_A-12',
#                                      '-port', '5555',
#                                      '-port_out', '5556',
#                                      '-max_seq_len', 'NONE',
#                                      '-pooling_strategy', 'NONE',
#                                      '-mask_cls_sep'])
# server = BertServer(args)
# server.start()

class DNN:

    def __init__(self):
        self.logdir = '../summary'
        self.checking_point_dir = "../checkpoint_dir"
        self.epochs = 80
        self.batch_size = 500
        self.model_name = "dnn_" + str(self.batch_size)
        self.regularizer = 0.001
        self.label = 3
        self.seed = 2019
        self.N = 10
        self.vet_size = 768
        self.max_sen_len = 25
        self.lstm_hidden_size = 300
        self.hidden_dense = 512
        self.lr = 0.001

        with tf.name_scope("inputs"):
            self.target = tf.placeholder(tf.float32, shape=(None, self.label), name="target")
            self.feature = tf.placeholder(tf.float32, shape=(None, self.max_sen_len, self.vet_size), name="input")
            self.dropout = tf.placeholder(tf.float32, name="dropout")

    def mkdir(self, path):
        """
        if the path doesn't exist, make the dirs of its.
        :param path:
        :return:
        """
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

    def iterate_minibatches(self, x, y, batchsize, shuffle=False):
        """
        Get minibatches
        :param X_title:
        :param X_tag:
        :param y:
        :param batchsize:
        :param shuffle:
        :return:
        """
        assert len(x) == len(y)
        if shuffle:
            indices = np.arange(len(x))
            np.random.shuffle(indices)
        for start_idx in range(0, len(x) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield x[excerpt], y[excerpt]

    def dnn_model(self, feature_vet, label, label_one_hot):
        """
        build the dnn model
        :param feature_vet:
        :param label:
        :param label_one_hot: one hot
        :return:
        """
        # with tf.name_scope("embedding"):
        #     embeddings = tf.get_variable("word_embeddings", [self.voc_size, self.embed_size])
        #     prefix_feature = tf.reshape(tf.gather(embeddings, self.prefix), (-1, self.embed_size))
        #     title_feature = tf.reshape(tf.gather(embeddings, self.title), (-1, self.embed_size))
        #     feature = tf.concat([prefix_feature, title_feature, self.tag], 1)

        with tf.name_scope("BiLSTM"):
            # bi-lstm
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_size)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_size)

            init_fw = lstm_fw_cell.zero_state(self.batch_size, dtype=tf.float32)
            init_bw = lstm_bw_cell.zero_state(self.batch_size, dtype=tf.float32)

            # weights = tf.get_variable("weights", [2 * self.hidden_size, self.output], dtype=tf.float32,  # 注意这里的维度
            #                           initializer=tf.random_normal_initializer(mean=0, stddev=1))
            # biases = tf.get_variable("biases", [n_classes], dtype=tf.float32,
            #                          initializer=tf.random_normal_initializer(mean=0, stddev=1))

            # outputs => [2, batch_size, max_time, depth]
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, \
                    self.feature, initial_state_fw=init_fw, initial_state_bw=init_bw)

            # concat the fw and bw
            fw_final_states = final_states[0].h
            bw_final_states = final_states[1].h
            fin_outputs = tf.concat((fw_final_states, bw_final_states), 1)
            # outputs = tf.transpose(outputs, [1, 0, 2])

        with tf.name_scope("dense"):
            dense = tf.layers.dense(inputs=fin_outputs, units=self.hidden_dense, activation=tf.nn.relu, \
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            drop_dense = tf.layers.dropout(dense, rate=self.dropout, training=True)
            y_pred = tf.layers.dense(inputs=drop_dense, units=self.label, activation=tf.nn.softmax, \
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))

        with tf.name_scope("Loss"):
            logloss = -tf.reduce_mean(tf.log(y_pred) * self.target) \
                      + tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', logloss)

        with tf.name_scope("training_op"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            training_op = optimizer.minimize(logloss)

        # Summary
        merged_summary = tf.summary.merge_all()
        self.mkdir(self.logdir)
        summary_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())

        # init
        init = tf.global_variables_initializer()

        # split train_data: train and val
        skf = StratifiedKFold(n_splits=self.N, random_state=self.seed, shuffle=True)
        for k, (train_in, test_in) in enumerate(skf.split(feature_vet, label)):
            x_train, x_val, y_train, y_val = feature_vet[train_in], feature_vet[test_in], label_one_hot[train_in], label_one_hot[test_in]
            break

        best_f1_score = 0
        best_epoch = 0
        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(self.epochs):
                start = time.time()
                # train
                loss_epoch = []
                for batch in self.iterate_minibatches(x_train, y_train, self.batch_size, shuffle=True):
                    x, y = batch
                    feed_dict_train = {
                        self.feature: x,
                        self.target: y,
                        self.dropout: 0.5,
                    }

                    sess.run(training_op, feed_dict_train)
                    e, summary = sess.run([logloss, merged_summary], feed_dict_train)
                    loss_epoch.append(e)

                summary_writer.add_summary(summary, epoch)
                print("epoch: %d" % epoch)
                print("train loss: %f" % np.mean(loss_epoch))

                # val:
                loss_epoch = []
                y_val_true = np.zeros((1, self.label))
                y_val_pre = np.zeros((1, self.label))
                for batch in self.iterate_minibatches(x_val, y_val, self.batch_size, shuffle=False):
                    x, y = batch
                    feed_dict_val = {
                        self.feature: x,
                        self.target: y,
                        self.dropout: 0.0,
                    }


                    e = logloss.eval(feed_dict=feed_dict_val)
                    loss_epoch.append(e)
                    ypred_val = sess.run(y_pred, feed_dict=feed_dict_val)
                    y_val_true = np.concatenate((y_val_true, y), axis=0)
                    y_val_pre = np.concatenate((y_val_pre, ypred_val), axis=0)

                f1_sco = f1_score(np.argmax(y_val_true[1:], axis=1), np.argmax(y_val_pre[1:], axis=1), average='macro')
                print("val loss: %f" % np.mean(loss_epoch))
                print("f1_score: %f" % f1_sco)

                saver = tf.train.Saver()
                self.mkdir(self.checking_point_dir)
                saver.save(sess, os.path.join(self.checking_point_dir, "dnn"), global_step=epoch)
                print("time: %f s" % (time.time() - start))

                # get best f1_score
                if(best_f1_score < f1_sco):
                    best_f1_score = f1_sco
                    best_epoch = epoch

            print("best_f1_score: %f" % best_f1_score)
            print("best_epoch: %d" % best_epoch)


if __name__ == "__main__":
    # read the all data
    file_open = open("../data/train_data.csv", encoding="gbk")
    data = pd.read_csv(file_open)
    label = np.reshape(np.array(data['label']), (-1, 1)) + 1
    text = data['scontent']

    # get embedding
    bc = BertClient()
    feature_vet = bc.encode(text.to_list())

    # label to one_hot
    one_hot_ec = OneHotEncoder()
    label_one_hot = one_hot_ec.fit_transform(label)
    label_one_hot = label_one_hot.toarray()

    dnn = DNN()
    dnn.dnn_model(feature_vet, label, label_one_hot)

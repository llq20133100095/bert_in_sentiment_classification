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
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import os
import time
import numpy as np

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

        with tf.name_scope("inputs"):
            self.target = tf.placeholder(tf.float32, shape=(None, self.label), name="target")
            self.feature = tf.placeholder(tf.float32, shape=(None, self.vet_size), name="input")
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

        with tf.name_scope("dense"):
            dense = tf.layers.dense(inputs=self.feature, units=512, activation=tf.nn.relu, \
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            drop_dense = tf.layers.dropout(dense, rate=self.dropout, training=True)
            y_pred = tf.layers.dense(inputs=drop_dense, units=self.label, activation=tf.nn.softmax, \
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))

        with tf.name_scope("Loss"):
            logloss = -tf.reduce_mean(tf.log(y_pred) * self.target) \
                      + tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', logloss)

        with tf.name_scope("training_op"):
            optimizer = tf.train.AdamOptimizer()
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
                feed_dict_val = {
                    self.feature: x_val,
                    self.target: y_val,
                    self.dropout: 0.0,
                }

                e = logloss.eval(feed_dict=feed_dict_val)
                loss_epoch.append(e)
                ypred_val = sess.run(y_pred, feed_dict=feed_dict_val)

                f1_sco = f1_score(np.argmax(y_val, axis=1), np.argmax(ypred_val, axis=1), average='macro')
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

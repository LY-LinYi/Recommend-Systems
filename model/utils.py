import numpy as np
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def dataframe_to_example(row, feature_list, all_features):
    """
    
    :param row: dataframe的每行数据
    :param feature_list: 拆分特征列表
    :param all_features: 所有特征列表
    """
    if len(feature_list) == 3:
        dense_features, sparse_features, label = feature_list
        seq_dense_features, seq_sparse_features = [], []
        position_features, position_seq_features = [], []
    elif len(feature_list) == 5:
        dense_features, sparse_features, label, seq_dense_features, seq_sparse_features = feature_list
        position_features, position_seq_features = [], []
    elif len(feature_list) == 7:
        dense_features, sparse_features, label, seq_dense_features, seq_sparse_features, position_features, position_seq_features = feature_list

    feature = {}
    for col in all_features:
        if col in dense_features:
            feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[row[col]]))
        elif col in sparse_features:
            feature[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[row[col].encode('utf-8')]))
        elif len(seq_dense_features) > 0 and col in seq_dense_features:
            feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=row[col]))
        elif len(seq_sparse_features) > 0 and col in seq_sparse_features:
            feature[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[s.encode('utf-8') for s in row[col]]))
        elif len(position_features) > 0 and col in position_features:
            feature[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[row[col].encode('utf-8')]))
        elif len(position_seq_features) > 0 and col in position_seq_features:
            feature[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[s.encode('utf-8') for s in row[col]]))
        elif col in label and isinstance(row[col], int):
            feature[col] = tf.train.Feature(int64_list=tf.train.Int64List(value=[row[col]]))
        elif col in label and isinstance(row[col], float):
            feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[row[col]]))
        else:
            feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[row[col]]))
    example = tf.train.Example(features=tf.train.Feature(feature=feature))

    return example.SerializeToString()


def dataframe_to_tfrecord(df, tf_path, feature_list, all_features):
    """
    
    :param df: 保存的dataframe数据
    :param tf_path: 保存数据地址
    :param feature_list: 拆分特征列表
    :param all_features: 所有特征列表
    """
    # 将dataframe写入TFRecord
    with tf.io.TFRecordWriter(tf_path) as writer:
        for idx, row in df.iterrows():
            example = dataframe_to_example(row, feature_list, all_features)
            writer.write(example)


def multi_pool(df, n_pools, n_threads, save_path, feature_list, all_features):
    """
    
    :param df: 保存的dataframe数据
    :param n_pools: 并行处理个数
    :param n_threads: 存储文件个数
    :param save_path: 保存数据地址
    :param feature_list: 拆分特征列表
    :param all_features: 所有特征列表
    """
    path_list = []
    batch_size = len(df) // n_threads
    with Pool(processes=n_pools) as pool:
        for i in range(n_threads):
            if i < n_threads - 1:
                pool.apply_async(dataframe_to_tfrecord, args=(df.iloc[i * batch_size: (i+1) * batch_size],
                                                            save_path + "_{}.tfrecords".format(i),
                                                            feature_list,
                                                            all_features))
            else:
                pool.apply_async(dataframe_to_tfrecord, args=(df.iloc[i * batch_size:],
                                                            save_path + "_{}.tfrecords".format(i),
                                                            feature_list,
                                                            all_features))
            path_list.append(save_path + "_{}.tfrecords".format(i))
        
        # 等待所有进程完成, 关闭进程池
        pool.close()
        # join函数等待所有子进程结束，才会执行主进程后的代码
        pool.join()
    return path_list



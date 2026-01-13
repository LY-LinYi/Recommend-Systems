import numpy as np
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


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


def parse(serialized_example, feature_list, all_features):
    """
    
    :param serialized_example: 解析数据地址
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
    features_description = {}
    for col in all_features:
        if col in dense_features:
            features_description[col] = tf.io.FixedLenFeature([], tf.float32, default_value=0)
        elif col in sparse_features:
            features_description[col] = tf.io.FixedLenFeature([], tf.string, default_value="0")
        elif col in seq_dense_features:
            features_description[col] = tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        elif col in seq_sparse_features:
            features_description[col] = tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True)
        elif col in position_features:
            features_description[col] = tf.io.FixedLenFeature([], tf.string, default_value="0")
        elif col in position_seq_features:
            features_description[col] = tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True)
        elif col in label:
            features_description[col] = tf.io.FixedLenFeature([], tf.float32, default_value=0)
        else:
            features_description[col] = tf.io.FixedLenFeature([], tf.float32, default_value=0)
    parse_features = tf.io.parse_single_example(serialized_example, features_description)
    feature_dict = {}
    for col in all_features:
        if col not in label:
            feature_dict[col] = parse_features[col]
    label_dict = tuple([parse_features[label[i]] for i in range(len(label))]) if len(label) > 1 else parse_features[label[0]]

    return feature_dict, label_dict


def get_evaluate(df, true_label, pred_label, pred_proba, text):
    """
    
    :param df: dataframe数据
    :param true_label: 真实标签列名
    :param pred_label: 预测标签列名
    :param pred_proba: 预测概率列名
    :param text: 文本描述
    """
    y_true = df[true_label].tolist()
    pred_label = df[pred_label].tolist()
    pred_proba = df[pred_proba].tolist()
    print('结果 {}: '.format(text))
    print('acc:{:.6f}'.format(accuracy_score(y_true=y_true, y_pred=pred_label)))
    print('auc:{:.6f}'.format(roc_auc_score(y_true=y_true, y_score=pred_proba)))
    target_names = ['negative', 'positive']
    x = classification_report(y_true=y_true, y_pred=pred_label, target_names=target_names)
    print(str(x))

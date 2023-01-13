# -*- coding: utf-8 -*-
# @Time    : 2022/10/19 18:26
# @Author  : Calvin Ren
# @Email   : rqx12138@163.com
# @File    : Bi_Gram.py
import re


# 读取训练数据
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        sentence_data = []
        for line in f.readlines():
            line = re.findall(r'[\u4e00-\u9fa5]+', line)  # 提取中文
            for temp in line:
                if len(temp) > 1:
                    sentence_data.append(temp)
        print("load data successfully")
        return sentence_data


# 计算词频
def train(train_data):
    double_words_dict = {}
    for sentence in train_data:
        if len(sentence) > 0:
            for i in range(len(sentence) - 1):
                # 数据平滑
                double_words_dict[sentence[i] + sentence[i + 1]] = double_words_dict.get(sentence[i] + sentence[i + 1],
                                                                                         0) + 1

    return double_words_dict


# 预测下一个字
def predict(input_sentence, double_words_dict):
    if len(input_sentence) == 0:  # 如果输入为空，返回错误提示
        print("input sentence is empty")
        return
    else:
        predict_word = input_sentence[-1]  # 读取输入句子中最后一个字
        predict_dict = {}
        for word in double_words_dict.keys():  # 遍历所有的双字词
            if predict_word == word[0]:
                predict_dict[word] = double_words_dict[word]
        if len(predict_dict) == 0:
            print("can not predict")  # 如果没有找到下一个字，返回错误提示
            return
        sorted_res = sorted(predict_dict.items(), key=lambda x: x[1], reverse=True)  # 按概率值降序排序
        i = 1
        for key, _ in sorted_res:  # 打印前5个预测结果
            print('top', i, ' selection: ', key[1:])
            i += 1
            if i > 5:
                break
        return


if __name__ == '__main__':
    data = load_data('news.txt')  # 读取训练数据
    double_words = train(data)  # 训练
    input_words = input('input：')  # 输入
    while True:  # 循环预测
        predict(input_words, double_words)
        next_input = str(input_words + input('input：{}'.format(input_words)))
        input_words = next_input

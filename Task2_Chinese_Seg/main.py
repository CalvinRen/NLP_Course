# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 23:03
# @Author  : Calvin Ren
# @Email   : rqx12138@163.com
# @File    : main.py
import jieba


# 读取搜狗语料库
def load_file():
    dic = {}  # 利用字典存储词及其类型
    # GB2312解码打开
    with open('SogouLabDic.dic', 'r', encoding='gb2312', errors='ignore') as f:
        file = f.readlines()
        for line in file:
            word = line.split('	')[0]
            word_type = line.split('	')[2].replace('\n', '').split(',')[:-1]
            dic[word] = word_type
    return dic


# 前向匹配算法
class fmm:
    def __init__(self, dic):
        self.dic = dic  # 读取词典

    # 分词函数
    def cut(self, sentence):
        result = []  # 存储分词结果
        type_result = []  # 存储分词类型
        while sentence:
            for i in range(len(sentence), 0, -1):
                word = sentence[:i]
                if word in self.dic:
                    result.append(word)
                    type_result.append(self.dic[word])
                    sentence = sentence[i:]  # 前向匹配，取剩余字符串
                    break
                # 如果最后一个字匹配不到，单字拆分
                if i == 1:
                    result.append(word)
                    type_result.append('?')  # 未知类型
                    sentence = sentence[i:]
        return result, type_result


# 后向匹配算法
class bmm:
    def __init__(self, dic):
        self.dic = dic  # 读取词典

    # 分词函数
    def cut(self, sentence):
        result = []  # 存储分词结果
        type_result = []  # 存储分词类型
        while sentence:
            for i in range(1, len(sentence) + 1):
                word = sentence[-i:]  # 后向匹配，取最后i个字
                if word in self.dic:
                    result.append(word)
                    type_result.append(self.dic[word])
                    sentence = sentence[:-i]  # 后向匹配，取剩余字符串
                    break
                # 如果最后一个字匹配不到，单字拆分
                if i == 1:
                    result.append(word)
                    type_result.append('?')  # 未知类型
                    sentence = sentence[:-i]
        return result[::-1], type_result[::-1]  # 反转


# 格式化输出
def format_print(result, type_result):
    res0 = ''
    # res1 = ''  # 词性输出
    for i in range(len(result)):
        res0 += str(result[i]) + ' / '
        # if type_result[i]:
        #     res1 += str(type_result[i][0]) + ' / '
        # else:
        #     res1 += '? / '
    print(res0)
    # print(res1)


# 双向匹配算法
def mm(fmm, bmm):
    # 如果分词结果词数相同
    if len(fmm) == len(bmm):
        fmm_count, bmm_count = 0, 0  # 统计单字数量
        for i in range(len(fmm)):
            if len(fmm[i]) == 1:
                fmm_count += 1
            if len(bmm[i]) == 1:
                bmm_count += 1
        # 分词结果相同，就说明没有歧义，可返回任意一个
        if fmm_count == bmm_count:
            return fmm
        # 分词结果不同，返回其中单宇较少的那个
        else:
            return fmm if fmm_count < bmm_count else bmm
    # 如果分词结果词数不同，返回其中单字较少的那个
    else:
        return fmm if len(fmm) < len(bmm) else bmm


if __name__ == '__main__':
    dic = load_file()  # 读取词典
    print("Loaded Success")
    fmm = fmm(dic)  # 前向匹配
    bmm = bmm(dic)  # 后向匹配
    sentence = '今天天气真好啊，我想出去玩'
    print("start cutting")
    fmm_res, _ = fmm.cut(sentence)  # 前向匹配分词结果
    bmm_res, _ = bmm.cut(sentence)  # 后向匹配分词结果
    print("================== cutting finished ==================")
    print("前向匹配结果: ")
    format_print(fmm_res, _)
    print("后向匹配结果: ")
    format_print(bmm_res, _)
    print()
    print("双向匹配匹配结果: ")
    format_print(mm(fmm_res, bmm_res), _)
    print("======================================================")
    # 使用jieba库生成分词结果，作为参考
    print("参考结果: ")
    seg_list = jieba.cut(sentence, cut_all=False)
    print(" / ".join(seg_list))

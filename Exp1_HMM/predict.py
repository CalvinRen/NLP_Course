import re
import pypinyin
import json


def load_testing_set(file):
    file = open(file, encoding='gb2312')
    test_py = []
    test_res = []
    line_num = 0
    for line in file.readlines():
        if line_num % 2 == 0:
            test_py.append(line)
        else:
            test_res.append(line)
        line_num += 1
    file.close()
    return test_py, test_res


def hz2py(hz):
    py = pypinyin.lazy_pinyin(hz)
    return py


def load_training_set(file):
    file = open(file, encoding='utf-8')
    training_txt = []
    i = 0
    for line in file.readlines():
        i += 1
        print('loading %d', i)
        tmp = re.findall(r'[\u4e00-\u9fa5]+', line)
        for word in tmp:
            if len(word) > 1:
                training_txt.append(word)
        # if tmp and tmp[0] and len(tmp[0]) > 1:
        #     training_txt.append(tmp[0])
    file.close()
    return training_txt


class hmm:
    def __init__(self):
        self.init_prob = {}
        self.trans_prob = {}
        self.emit_prob = {}
        self.py2hz_dic = {}

    def train_py2hz(self, file):
        file = open(file, encoding='utf-8-sig')
        py_dic = {}
        for line in file:
            line = line.strip().split()
            py_dic[line[0]] = line[1]
        self.py2hz_dic = py_dic
        file.close()

    def train_init_trans_prob(self, file):
        single_words = {}
        double_words = {}
        for words in file:
            for i in range(len(words)):
                single_words[words[i]] = single_words.get(words[i], 0) + 1
                if i != 0:
                    double_words[words[i - 1:i + 1]] = double_words.get(words[i - 1:i + 1], 0) + 1
        for word in single_words.keys():
            self.init_prob[word] = single_words[word] / sum(single_words.values())

        for word in double_words.keys():
            self.trans_prob[word] = double_words[word] / single_words[word[0]]

    def train_emit_prob(self, file):
        for line in file:
            for words in line:
                py_res = hz2py(words)
                for i in range(len(py_res)):
                    self.emit_prob[py_res[i]] = self.emit_prob.get(py_res[i], {})
                    self.emit_prob[py_res[i]][words[i]] = self.emit_prob[py_res[i]].get(words[i], 0) + 1
        for py in self.emit_prob.keys():
            for word in self.emit_prob[py].keys():
                self.emit_prob[py][word] = self.emit_prob[py][word] / sum(self.emit_prob[py].values())

    def viterbi(self, word_lst):
        word_lst = word_lst.strip().split()
        word_lst = [word.lower() for word in word_lst]
        n = len(word_lst)
        dp = {}

        for i in range(n):
            dp[i] = {}
        for i in range(len(self.py2hz_dic[word_lst[0]])):

            if self.py2hz_dic[word_lst[0]][i] in self.init_prob.keys() and self.py2hz_dic[word_lst[0]][i] in self.emit_prob[word_lst[0]].keys():
                dp[0][self.py2hz_dic[word_lst[0]][i]] = self.init_prob[self.py2hz_dic[word_lst[0]][i]] * self.emit_prob[word_lst[0]][self.py2hz_dic[word_lst[0]][i]]
            elif self.py2hz_dic[word_lst[0]][i] in self.init_prob.keys():
                dp[0][self.py2hz_dic[word_lst[0]][i]] = self.init_prob[self.py2hz_dic[word_lst[0]][i]]
            else:
                dp[0][self.py2hz_dic[word_lst[0]][i]] = 0
        for i in range(1, n):
            for j in range(len(self.py2hz_dic[word_lst[i]])):
                dp[i][self.py2hz_dic[word_lst[i]][j]] = 0
                if self.py2hz_dic[word_lst[i]][j] in self.emit_prob[word_lst[i]].keys():
                    emit_pro = self.emit_prob[word_lst[i]][self.py2hz_dic[word_lst[i]][j]]
                else:
                    emit_pro = 0

                for k in range(len(self.py2hz_dic[word_lst[i - 1]])):
                    trans_tmp = self.py2hz_dic[word_lst[i-1]][k] + self.py2hz_dic[word_lst[i]][j]
                    if trans_tmp in self.trans_prob.keys():
                        trans_pro = self.trans_prob[trans_tmp]
                    else:
                        trans_pro = 0
                    dp[i][self.py2hz_dic[word_lst[i]][j]] = max(dp[i-1][self.py2hz_dic[word_lst[i-1]][k]] * trans_pro * emit_pro, dp[i][self.py2hz_dic[word_lst[i]][j]])

        # 列出最大值
        res_path = []
        max_tmp = -1
        max_key = None
        for i in dp[n-1].keys():
            if max_tmp < dp[n-1][i]:
                max_tmp = dp[n-1][i]
                max_key = i
        res_path.append(max_key)

        pre_key = max_key
        for i in range(n-2, -1, -1):
            max_value = -1
            for j in dp[i].keys():
                if j+pre_key in self.trans_prob.keys():
                    trans_p = self.trans_prob[j+pre_key]
                else:
                    trans_p = 0
                if pre_key in self.emit_prob[word_lst[i+1]].keys():
                    emit_p = self.emit_prob[word_lst[i+1]][pre_key]
                else:
                    emit_p = 0
                if dp[i][j] * trans_p * emit_p > max_value:
                    max_value = dp[i][j]
                    max_key = j
            # print('t', self.trans_prob[max_key+pre_key])
            # print('e', self.emit_prob[word_lst[i+1]][pre_key])
            res_path.append(max_key)
            pre_key = max_key
        res_path.reverse()

        # for i in range(n):
        #     max_key = None
        #     max_value = -1
        #     for j in dp[i].keys():
        #         if max_value < dp[i][j]:
        #             max_value = dp[i][j]
        #             max_key = j
        #     res_path.append(max_key)
        return res_path

    def accuracy(self, py_file, res_file):
        print("测试开始")
        test_py, test_res = py_file, res_file
        print("测试集加载完成")
        n = len(test_py)
        avg_acc = 0
        for i in range(n):
            predict_res = self.viterbi(test_py[i])
            if i != n-1:
                print("第{}个测试样本的原句为：{}".format(i, str.join('', test_res[i])), end='')
            else:
                print("第{}个测试样本的原句为：{}".format(i, str.join('', test_res[i])))
            print("第{}个测试样本的预测结果为：{}".format(i, str.join('', predict_res)))
            acc_count = 0
            for j in range(len(predict_res)):
                if predict_res[j] == test_res[i][j]:
                    acc_count += 1
            print("第{}个测试样本的准确率为：{:.2f}%".format(i, acc_count / len(predict_res) * 100))
            avg_acc += acc_count / len(predict_res)
            print()  # 换行
        print("平均准确率为：{:.2f}%".format(avg_acc / n * 100))
        print("测试结束")

    def train(self):
        print("训练开始")
        tran_dataset = load_training_set(r'toutiao_cat_data.txt')
        self.train_py2hz(r'pinyin2hanzi.txt')
        self.train_init_trans_prob(tran_dataset)
        self.train_emit_prob(tran_dataset)
        init_prob_json = json.dumps(self.init_prob, ensure_ascii=False)
        trans_prob_json = json.dumps(self.trans_prob, ensure_ascii=False)
        emit_prob_json = json.dumps(self.emit_prob, ensure_ascii=False)
        py2hz_json = json.dumps(self.py2hz_dic, ensure_ascii=False)
        with open('json/init_prob.json', 'w+') as file:
            file.write(init_prob_json)
        with open('json/trans_prob.json', 'w+') as file:
            file.write(trans_prob_json)
        with open('json/emit_prob.json', 'w+') as file:
            file.write(emit_prob_json)
        with open('json/py2hz.json', 'w+') as file:
            file.write(py2hz_json)
        print("训练集加载完成")

    def load(self):
        with open('json/init_prob.json', 'r+') as file:
            content = file.read()
            content = json.loads(content)
            self.init_prob = content
        with open('json/trans_prob.json', 'r+') as file:
            content = file.read()
            content = json.loads(content)
            self.trans_prob = content
        with open('json/emit_prob.json', 'r+') as file:
            content = file.read()
            content = json.loads(content)
            self.emit_prob = content
        with open('json/py2hz.json', 'r+') as file:
            content = file.read()
            content = json.loads(content)
            self.py2hz_dic = content
        print("模型加载完成")


def main():
    test_hmm = hmm()
    # print(load_training_set(r'toutiao_cat_data.txt'))
    # test_hmm.train()
    test_hmm.load()
    test_py, test_res = load_testing_set(r'测试集.txt')
    test_hmm.accuracy(test_py, test_res)


if __name__ == '__main__':
    main()

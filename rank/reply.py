import math
import time
import re
import os
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import unicodedata
import pickle
from collections import defaultdict, Counter

class SentenceRank(object):
    """句子排序
    """
    def __init__(self):
        self.root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        data_frame = pd.read_csv(os.path.join(self.root_path, "comments.csv"))
        self.sentences = data_frame['text'].values
        self.sentences = self._split_str(self.sentences)

        self.sentences = map(self._preprocess, self.sentences)
        self.sentences = list(filter(lambda s: s !='', self.sentences))
        
        s = " ".join(self.sentences)
        for x, w in jieba.analyse.extract_tags(s, topK=10, withWeight=True):
            print('%s %s' % (x, w))


        self.sens_ws = self.split_word(self.sentences)
        print("\033[0;34mLoad word2vec\033[0m") 
        self.word_vec = self.load_word2vec()
        self.word_vec = self.create_matrix(self.word_vec)

        self.sens_vec = self.sens2vec(self.sens_ws, self.word_vec)
        self.sim_matrix = self.sim_matrix(self.sens_vec)
        print("\033[0;34mInit finish\033[0m") 
    def sen_rank(self, d=0.85):
        ws = defaultdict(float)
        outSum = defaultdict(float)
        
        wsdef = 1.0 / (float(self.sim_matrix.shape[0]) or 1.0)
        for n, out in enumerate(self.sim_matrix):
            ws[n] = wsdef
            outSum[n] = np.sum(out)
        for x in range(10):
            for n, out in enumerate(self.sim_matrix):
                s = 0
                for i, e in enumerate(out):
                    s += e / outSum[i] * ws[i]
            ws[n] = (1 - d) + d * s
        ws = sorted(ws.items(), key=lambda key: key[1], reverse=True)
        return ws

    def sim_matrix(self, sens_vec, Threshold=0.0):
        """创建rank矩阵
        Args:
            sens_vec: (numpy.array)句子
        Returns:
            sim_matrix: (numpy.array)
        """
        sen_len = sens_vec.shape[0]
        sim_matrix = np.zeros(shape=(sen_len, sen_len), dtype=np.float64)
        
        for i in range(sen_len):
            for j in range(i, sen_len):
                if i == j:
                    sim_matrix[i][j] = 0.0
                    continue
                vec_i = sens_vec[i]
                vec_j = sens_vec[j]
                sim = self.distance_sim(vec_i, vec_j)
                if sim < Threshold:
                    continue
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim
        return sim_matrix
    def distance_sim(self, s1, s2):
        return np.sqrt(np.sum(np.square(s1 - s2)))
    def cos_sim(self, s1, s2):
        """计算两个句子的余弦值
        Args:
            s1: (numpy.array)句子的矩阵
            s2: (numpy.array)同上
        Returns:
            cos: (int)s1, s2的余弦
        """

        s1_m = np.linalg.norm(s1, ord=2)
        s2_m = np.linalg.norm(s2, ord=2)
        if s1_m == 0 or s2_m == 0:
            return 0.0
        cos = np.sum(s1 * s2)/(s1_m * s2_m)
        return np.abs(cos)
    def sens2vec(self, sentences, word_vec):
        """文本转vocab索引
        Args:
            s: (str)文本
        Returns:
            s: (numpy)文本矩阵
        """
        sens_vec = []
        for sentence in sentences:
            word_vecs = []
            for word in sentence:
                if word in word_vec:
                    word_vecs.append(word_vec[word])
            word_vecs = np.array(word_vecs, dtype=np.float64)
            s_vec = np.mean(word_vecs, axis=0, dtype=np.float64)
            sens_vec.append(s_vec)
        return np.array(sens_vec)
    def create_matrix(self, word_vec):
        count = Counter(self.words).most_common()
        print(f"\033[0;35m频率前五的词\033[0;33m{count[:5]}\033[0m")
        word_dim = len(list(word_vec.values())[0])
        zero_vec = [0]*word_dim
        w_vec = {}
        for word, _ in count:
            if word in word_vec:
                w_vec[word] = word_vec[word]
                continue
            w_vec[word] = zero_vec
        return w_vec
        print(len(count))
    def split_word(self, sentences):
        self.words = []
        sens_list = []
        
        for sentence in sentences:
            words = jieba.cut(sentence)
            words = list(words)
            self.words.extend(words)
            sens_list.append(words)
        return sens_list
    def load_word2vec(self):
        """载入词向量
        Args:
            f_path: (str)词向量文件地址
        Returns:
            vocab_dict: key为vocab的value，value为vocab的value对应索引
            vocab: 词向量的词集合
            matrix: 词向量矩阵
        """
        f_name = 'sgns.weibo.bigram'
        f_path = os.path.join(self.root_path, 'ngram2vec', f_name)
        dirs = "word_vec"

        f_pkl = os.path.join(self.root_path, dirs, f"{f_name}.pkl")
        if os.path.exists(f_pkl):
            word_vec = np.load(f_pkl, allow_pickle=True) 
            return word_vec
        
        if not os.path.exists(os.path.join(self.root_path, dirs)):
            os.makedirs(os.path.join(self.root_path, dirs))
        with open(f_path, errors='ignore') as f:
            word_vec = {}
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    continue
                line = line.rstrip().split(' ')
                word = line[0]
                vec = line[1:]
                word_vec[word] = vec
            with open(f_pkl, 'wb') as fo:
                pickle.dump(word_vec, fo)
            return word_vec
    def exit(self):
        del self.sentences
        del self.sens_ws
        del self.word_vec
        del self.sens_vec
        del self.sim_matrix
    def _split_str(self, sentences):
        sens_flat = []
        for sentence in sentences:
            sens_list = re.split(r"(。|\.|！｜\!|;|；|,|，)", str(sentence))
            sens_flat.extend(sens_list)
        return sens_flat
    def _preprocess(self, s):
        w = self._unicode_to_ascii(s)
        w = re.sub(r'[" "]+', "", s)
        w = re.sub(r"[^\u4e00-\u9fa5]+", "", s)
        return w.rstrip().strip()
    def _unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

if __name__ == '__main__':
    start_time = time.time()
    with  SentenceRank() as sen_rank:
        tags = sen_rank.sen_rank()
        for index in range(10):
            print(f"\033[0;36m{sen_rank.sentences[tags[index][0]]}\033[0m\n")
    end_time = time.time()
    print("\033[0;34mtotal time:\033[0m", end_time - start_time)

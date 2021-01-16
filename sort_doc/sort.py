import time
import re
import os
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import unicodedata
import pickle
from collections import Counter

# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import tensorflow as tf

class Sort(object):
    """句子排序
    """
    
    def __init__(self, sentences, f_path, output_path='word_vec'):
        self.sentences = []
        self.sens_index = []

        self._split_str(sentences)
        self.sentences = map(self._preprocess, self.sentences)
        self._filter(self.sentences)
#        self.sentences = list(filter(lambda s: s !='' and len(s)>1, self.sentences))
#        self.sentences = list(np.unique(self.sentences))
        
        print(f"\033[0;32m{next(iter(self.sentences))}\033[0m")
        print("\033[0;34mSentences set len:\033[0m", len(self.sentences)) 
        self.sens_ws = self.split_word(self.sentences)
        
        print(f"\033[0;34mLoad word2vec sens_index_len:{len(self.sens_index)},\
                senteces:{len(self.sentences)}\033[0m") 
        self.word_vec = self.load_word2vec(f_path, output_path)
        self.word_vec = self.create_matrix(self.word_vec)
        print("\033[0;34mInit finish\033[0m")
        self.sens_vec = self.sens2vec(self.sens_ws, self.word_vec) 
    def word_kmeans(self):
        words = list(self.word_vec.keys())
        vecs = np.array(list(self.word_vec.values()), dtype=np.float64)
        y_pred = self.k_means(vecs)
        
#        num = 40
#        p_words = words[:num]
#        p_vecs = vecs[:num]
#        p_y_pred = y_pred[:num]
#
#        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
#        fig, axs = plt.subplots(1, 1, figsize=(16, 8), sharey=True)
#        axs.scatter(p_vecs[:,0],p_vecs[:,1],c=p_y_pred)
#        for i in range(len(p_words)):    #给每个点进行标注
#            axs.text(p_vecs[:,0][i]*1.03,p_vecs[:,1][i]*1.03, p_words[i],
#            fontsize=12, color = "black", style = "italic", weight = "light",
#            verticalalignment='center', horizontalalignment='right',rotation=0)
#        fig.savefig('word_k_vecs.png')

        return words, y_pred
    def sen_kmeans(self):
        vecs = self.sens_vec
        y_pred = self.k_means(vecs)
        return self.sentences, self.sens_index, y_pred
    def k_means(self, vecs, n_clusters=None):
        if n_clusters is None:
            bandwidth = estimate_bandwidth(vecs, quantile=0.2)

            ms = MeanShift(bandwidth=bandwidth)
            ms.fit(vecs)
            labels = ms.labels_
            labels_unique = np.unique(labels)
            n_clusters = len(labels_unique)
            print(f"\033[0;35mnumber of estimated clusters : \033[0;36m{n_clusters}\033[0m")
        return KMeans(n_clusters=n_clusters).fit_predict(vecs)
    def word_rank(self, batch_size=64, iter_num=10, d=0.85):
        words = list(self.word_vec.keys())
        vecs = np.array(list(self.word_vec.values()), dtype=np.float64)
        vecs = np.expand_dims(vecs, axis=1)
        sim_matrix = self.batch_create_sim_matrix(vecs, batch_size)
        ranks = self.calculate_rank(sim_matrix, iter_num=iter_num, d=d).numpy()
        
        return words, ranks.flatten()
    def sen_rank(self, batch_size=64, iter_num=10, d=0.85):
        vecs = np.expand_dims(self.sens_vec, axis=1)
        sim_matrix = self.batch_create_sim_matrix(vecs, batch_size)
        ranks = self.calculate_rank(sim_matrix, iter_num=iter_num, d=d).numpy()
        
        return self.sentences, self.sens_index, ranks.flatten()
    def batch_create_sim_matrix(self, vecs, batch_size):
        vecs = tf.data.Dataset.from_tensor_slices(vecs).batch(batch_size)
        sim_matrix = []
        for vecs_raw in vecs:
            matrix_raw = []
            for vecs_col in vecs:
                sims = self.create_sim_matrix(vecs_raw, vecs_col)
                if len(matrix_raw) == 0:
                    matrix_raw = sims
                    continue
                matrix_raw = tf.concat([matrix_raw, sims], axis=1)
            if len(sim_matrix) == 0:
                sim_matrix = matrix_raw
                continue
            sim_matrix = tf.concat([sim_matrix, matrix_raw], axis=0)
        return sim_matrix
    @tf.function
    def calculate_rank(self, matrix, iter_num, d):
        ws = tf.ones((matrix.shape[0], 1), dtype=tf.float64)
        out_sum = tf.math.reduce_sum(matrix, axis=-1, keepdims=True)
        mul_mat = matrix/out_sum

        for x in range(iter_num):
            ws = (1 - d) + d * tf.linalg.matmul(mul_mat, ws, transpose_a=True)
        return ws
    @tf.function
    def create_sim_matrix(self, vecs_raw, vecs_col):
        """创建rank矩阵
        Args:
            vecs: (numpy.array)句子
        Returns:
            sim_matrix: (numpy.array)
        """
        matrix = vecs_raw - tf.transpose(vecs_col, perm=[1,0,2])
        matrix = tf.math.square(matrix)
        matrix = tf.math.reduce_sum(matrix, axis=-1)
        matrix = tf.math.sqrt(matrix)
        reduce_max = tf.math.reduce_max(matrix)
        return reduce_max - matrix
    def sens2vec(self, sentences, word_vec):
        """文本转vocab索引
        Args:
            s: (str)文本
        Returns:
            s: (numpy)文本矩阵
        """
        print(f"\033[0;32m{next(iter(sentences))}\033[0m") 
        sens_vec = []
        sens_filter = []
        sens_index = []
        for index, sentence in enumerate(sentences):
            word_vecs = []
            for word in sentence:
                if word in word_vec:
                    word_vecs.append(word_vec[word])
            if len(word_vecs) == 0:
                continue
            sens_filter.append(self.sentences[index])
            sens_index.append(self.sens_index[index])

            word_vecs = np.array(word_vecs, dtype=np.float64)
            word_vecs_mean = np.mean(word_vecs, axis=0, dtype=np.float64)
            sens_vec.append(word_vecs_mean)
        
        self.sentences = sens_filter
        self.sens_index = sens_index
        print("\033[0;34mSentences set len:\033[0m", len(self.sentences)) 
        return np.array(sens_vec)
    def create_matrix(self, word_vec):
        count = Counter(self.words).most_common()
        print(f"\033[0;35m总共有:{len(count)}频率前五的词\033[0;33m{count[:5]}\033[0m")
        w_vec = {}
        for word, _ in count:
            if word in word_vec:
                w_vec[word] = word_vec[word]
        return w_vec
    def split_word(self, sentences):
        self.words = []
        sens_list = []
        for sentence in sentences:
            words = jieba.cut(sentence)
            words = list(words)
            self.words.extend(words)
            sens_list.append(words)
        return sens_list
    def load_word2vec(self, f_path, f_pkl):
        """载入词向量
        Args:
            f_path: (str)词向量文件地址
        Returns:
            vocab_dict: key为vocab的value，value为vocab的value对应索引
            vocab: 词向量的词集合
            matrix: 词向量矩阵
        """

        if os.path.exists(f_pkl):
            word_vec = np.load(f_pkl, allow_pickle=True) 
            return word_vec
        
        #if not os.path.exists(os.path.join(root_path, output_dir)):
        #    os.makedirs(os.path.join(root_path, output_dir))
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
    def _filter(self, sentences):
        sens_filter = []
        sens_index = []
        for index, sentence in enumerate(sentences):
            if sentence == '' or len(sentence) <= 1:
                continue
            sens_filter.append(sentence)
            sens_index.append(self.sens_index[index])
        self.sens_index = sens_index
        self.sentences = sens_filter
    def _unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    def _preprocess(self, s):
        s = self._unicode_to_ascii(s)
        s = re.sub(r'[" "]+', "", s)
        s = re.sub(r"[^\u4e00-\u9fa5]+", "", s)
        s = re.sub(r"(的|了|和|是|就|都|而|及|與|著|或|一個|沒有|我們|你們|妳們|他們|她們|是否)", "", s)
        return s.rstrip().strip()
    def _split_str(self, sentences):
        for index, sentence in enumerate(sentences):
            sens_list = re.split(r"(\s|？|\?|。|\.|！|\!|;|；|,|，|:|：)", str(sentence))
            self.sens_index.extend([index]*len(sens_list))
            self.sentences.extend(sens_list)
    def exit(self):
        del self.sentences
        del self.sens_ws
        del self.word_vec
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

if __name__ == '__main__':
    start_time = time.time()
    
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_frame = pd.read_csv(os.path.join(root_path, "comments.csv"))
    word2vec_path = os.path.join(root_path, 'ngram2vec', 
        'sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5')
    output_word_vec = os.path.join(root_path, 'word_vec',
        'sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.pkl')
    sentences = data_frame['text'].values
     
    with Sort(sentences, word2vec_path, output_word_vec) as sort:
        sens, sens_index, ranks = sort.sen_rank()
        sen_ranks = list(zip(sens, ranks, sens_index))
        sen_ranks = sorted(sen_ranks, key=lambda x: x[1], reverse=True)
        for index in range(20):
            print(f"\033[0;36m{sen_ranks[index][0]}\033[0;35m{sentences[sen_ranks[index][2]]}\033[0m\n")
#        _, sorts, _ = sort.sen_kmeans()
#        sen_sort = list(zip(sentences, sorts, sens_index))
#        for index in range(20):
#            print(f"\033[0;36m{sen_sort[index][0]}\033[0;35m{sentences[sen_sort[index][2]]}\033[0m\n")
    end_time = time.time()
    print("\033[0;34mtotal time:\033[0m", end_time - start_time)

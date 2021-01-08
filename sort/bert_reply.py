import os
import time
import re
import unicodedata
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def bert_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor_path = os.path.join(root_path, 'bert_zh', 'bert_zh_preprocess_2')
    preprocessor = hub.KerasLayer(preprocessor_path, trainable=False)
    model_path = os.path.join(root_path, 'bert_zh', 'bert_zh_L-12_H-768_A-12_3')
    encoder = hub.KerasLayer(model_path, trainable=False)
    encoder_inputs = preprocessor(text_input)
    outputs = encoder(encoder_inputs)['pooled_output']
    return tf.keras.Model(inputs=text_input, outputs=outputs)

class SentenceRank(object):
    """
    """
    def __init__(self):
        root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        data_frame = pd.read_csv(os.path.join(root_path, 'comments.csv'))
        self.sentences = data_frame['text'].values
        del data_frame
        self.sentences = self._split_str(self.sentences)
        self.sentences = map(self._preprocess, self.sentences)
        self.sentences = list(filter(lambda s: s != '', self.sentences))
        
        self.b_model = bert_model() 
        self.sens_vec = self.run(self.sentences)
        print(self.sens_vec)
    def run(self,sentences):
        dataset = tf.data.Dataset.from_tensor_slices(sentences).batch(64)
        predicts = []
        for x in dataset:
            predicts.append(self.predict(x).numpy())
        return predicts
    @tf.function
    def predict(self, x):
        return self.b_model(x)
    def _unicode_to_ascii(self, sentence):
        return ''.join(
            c for c in unicodedata.normalize('NFD', sentence) if unicodedata.category(c) != 'Mn')
    def _preprocess(self, sentence):
        sentence = self._unicode_to_ascii(sentence)
        sentence = re.sub(r'[" "]+', "", sentence)
        sentence = re.sub(r"[^\u4e00-\u9fa5]+", "", sentence)
        return sentence.rstrip().strip()
    def _split_str(self, sentences):
        sens_flat = []
        for sentence in sentences:
            sens_list = re.split(r"(。|\.|！|\!|;|；|,|，)", str(sentence))
            sens_flat.extend(sens_list)
        return sens_flat
    def exit(self):
        del self.sentences
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

if __name__ == '__main__':
    start_time = time.time()
    with SentenceRank() as sen_rank:
        pass
    end_time = time.time()
    print(f"\033[0;35mTotal time:{end_time-start_time}\033[0m")

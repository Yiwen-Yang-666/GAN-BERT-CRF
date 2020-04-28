
import numpy as np
import tensorflow as tf
from bert_serving.client import BertClient
params = {
    'dim': 200,
    'dropout': 0.5,
    'num_oov_buckets': 1,
   # 'batch_size': 20,
    'buffer': 15000,
    'lstm_size': 100,
    'words': '../../../medical_char_data_cleaned/vocab.words.txt',
    'chars': '../../../medical_char_data_cleaned/vocab.chars.txt',
    'tags': '../../../medical_char_data_cleaned/vocab.tags.txt',
    'glove': '../../../medical_char_data_cleaned/glove.npz'
   #  from_detection_checkpoint: true
}

glove = np.load(params['glove'])['embeddings']  # np.array
with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    variable1 = np.vstack([glove, [[0.] * params['dim']]])

x = [['student','have','books'],['having', 'a', 'good', 'day']]
vocab_words = tf.contrib.lookup.index_table_from_file(params['words'], num_oov_buckets=params['num_oov_buckets'])
print(vocab_words)

word_ls = []
with open(params['words'], 'r', encoding='utf-8') as rd:
    for line in rd.readlines():
        word_ls.append(line.strip())

print(word_ls)
bc = BertClient()
m = bc.encode(['First do it', 'then do it right', 'then do it better', '0 0 0'])

all_word_vec = bc.encode(word_ls[1:20])


embeddings = np.zeros([len(word_ls), 768])
print(type(m), m.shape)
print(all_word_vec.shape)
for ind,word in enumerate(word_ls):
    if ind % 1000 == 0:
        print('计算了{}个'.format(ind))
    vec = bc.encode([word])
    embeddings[ind] = vec

np.savez_compressed('bert_vec.npz', embeddings=embeddings)
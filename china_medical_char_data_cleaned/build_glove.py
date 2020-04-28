"""Build an np.array from some glove file and some vocab file

You need to download `glove.840B.300d.txt` from
https://nlp.stanford.edu/projects/glove/ and you need to have built
your vocabulary first (Maybe using `build_vocab.py`)
"""

__author__ = "Guillaume Genthial"

from pathlib import Path

import numpy as np
import codecs

if __name__ == '__main__':
    # Load vocab
    with Path('vocab.words.txt').open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    #print('word_to_idx',word_to_idx)
    size_vocab = len(word_to_idx)
    #print('size_vocab',size_vocab)
    # Array of zeros
    embeddings = np.zeros([size_vocab, 200])

    # Get relevant glove vectors
    found = 0
    #print('Reading GloVe file (may take a while)')
    #with Path('glove.840B.300d.txt').open() as f:
    with codecs.open('char_word2vec_clean_200.utf8','a+','utf8') as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 200 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                #print('word_idx',word_idx)
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed('glove.npz', embeddings=embeddings)

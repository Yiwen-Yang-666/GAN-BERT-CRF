# encoding=utf8
"""Script to build words, chars and tags vocab"""

__author__ = "Guillaume Genthial"

from collections import Counter
from pathlib import Path
import codecs

# TODO: modify this depending on your needs (1 will work just fine)
# You might also want to be more clever about your vocab and intersect
# the GloVe vocab with your dataset vocab, etc. You figure it out ;)
MINCOUNT = 1

if __name__ == '__main__':
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save
    def words(name):
        return '{}.words.txt'.format(name)

    print('Build vocab words (may take a while)')
    counter_words = Counter()
    for n in ['train', 'testa','testb']:
        with Path(words(n)).open() as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}
    #print(vocab_words)
    #with Path('vocab.words.txt').open('w') as f:
    with codecs.open('vocab.words.txt','a+','utf8') as f:
        for w in sorted(list(vocab_words)):
            #print('w',type(w))
            f.write(w+'\n')
    print('- done. Kept {} out of {}'.format(
        len(vocab_words), len(counter_words)))

    # 2. Chars
    # Get all the characters from the vocab words
    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    #with Path('vocab.chars.txt').open('w') as f:
    with codecs.open('vocab.chars.txt','a+','utf8') as f:
        for c in sorted(list(vocab_chars)):
            f.write(c+'\n')
    print('- done. Found {} chars'.format(len(vocab_chars)))

    # 3. Tags
    # Get all tags from the training set

    def tags(name):
        return '{}.tags.txt'.format(name)

    print('Build vocab tags (may take a while)')
    vocab_tags = set()
    #with Path(tags('train')).open() as f:
    for n in ['train', 'testa','testb']:
        with codecs.open(tags(n),'r','utf8') as f:
            for line in f:
                vocab_tags.update(line.strip().split())

    #with Path('vocab.tags.txt').open('w') as f:
    with codecs.open('vocab.tags.txt','a+','utf8') as f:
        for t in sorted(list(vocab_tags)):
            f.write(t+'\n')
    print('- done. Found {} tags.'.format(len(vocab_tags)))

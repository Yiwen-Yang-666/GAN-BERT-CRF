#!/bin/bash
# contains datacheck, data transfor, build vocab and build glove four parts
echo 'delete existed original training file'
rm *.txt
echo 'check training data form!'
python check_training_data.py train_data
echo 'transform training data!'
python data_transfor.py train_data
echo 'build vocabulary'
python build_vocab.py
echo 'build glove.py file'
python build_glove.py
echo 'finished data initial'

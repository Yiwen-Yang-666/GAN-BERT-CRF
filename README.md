# GAN-BERT-CRF AND GAN-BiLSTM-CRF
#### This is an algorithm which uses BERT-CRF as the generator and a CNN-based network as the discrminator. This algorithm can use both annotated samples and unannotated samples to maximize NER (i.e Name entity recognition) performance. Furthermore, it plays a role of active learning, and can effectively identify sequence samples with error labels. Thus, it has three features. The paper for this algorithm would be published soon.
* NER: Use both unannotated and annotated samples to maximize NER performances.
* Active leanring: Use both ouputs of CRF layer and the discriminator to select unannotated samples to be labeled.
* Identification of error labels: The discriminator outputs smaller score for the sequences with error labels.

### Requirements
* Python: 3.6
* Tensorflow: 1.15.2

### DataSet
The DataSet should be in the format of two columns. The words are in the first column and labels are in the second column. Sample data is in the Data Directory.<br>
The china_medical_char_data_cleaned directory processes Chinese data and medical_char_data_cleaned processes English data. First, Download dataset and put dataset into these two directories based on languages. Then put your embedding files into these dicrectories and  run .sh file to produce training data, validation data, test data and embedding npz.
```
bash makefile.sh
```

### Usage
Download pretrained Bert and divide you training data into two parts: labeled data and unlabeled data. Put your divided training data、validation data and test data into model/data_path. (pretrained Bert: https://github.com/google-research/bert)<br>
Train data
```
python main.py --mode=train
```

### How to optimize
One of effective optimize methods is to update different layers of BERT, you can set up which layers are updated in the bert_gan file.For example, 
```
self.params_1 = [param for param in tf.trainable_variables() if
                         ('crf_param' not in param.name and 'bert/encoder/layer_11/output' in param.name) ]
grads_and_vars2 = optim.compute_gradients(self.loss2, self.params_1)
```
      

Test data and carry out active learning steps

```
python main.py --mode=test
```

### Reference
[BERT] (https://github.com/google-research/bert)<br>
(https://github.com/guillaumegenthial/sequence_tagging)



.

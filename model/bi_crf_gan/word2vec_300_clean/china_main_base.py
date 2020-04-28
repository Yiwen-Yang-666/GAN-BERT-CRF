import numpy as np
import random
import tensorflow as tf
import random
from bert_crf import Generator_BiLSTM_CRF
import numpy as np
import os, time, sys, argparse
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import batch_yield, read_corpus, pad_sequences, batch_yield_for_unla_da, read_corpus_unlabel, \
    batch_yield_for_discri, batch_yield_for_discri_unlabeled
# from discriminator_for_gram import Discriminator
from bert_base.bert import modeling

bert_path = '/home/ywd/tf_model/pre_training_model/chinese_L-12_H-768_A-12/'
init_checkpoint = os.path.join(bert_path, 'bert_model.ckpt')
#################################

# chinese data ccks
data_path = 'ccks_data_path'
# # english data 2010ib
# data_path = 'data_path'
#### Generator Hyper-parameters
batch_size = 20
epoch_num = 10
# filter_sizes = [1, 2, 3, 4, 5, 6]
filter_sizes = [1, 2, 3, 4]
num_filters = [100, 200, 200, 200]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default=data_path, help='train data source')
parser.add_argument('--train_data_unlabel', type=str, default=data_path, help='train data source')
parser.add_argument('--mode', type=str, default='train', help='train/test')
# parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
parser.add_argument('--test_data', type=str, default=data_path, help='test data source')
parser.add_argument('--sub_test_data', type=str, default=data_path, help='test data source')
args = parser.parse_args()
# train_data =
params = {
    'dim': 768,
    'dropout': 0.5,
    'num_oov_buckets': 1,
    # 'batch_size': 20,
    'buffer': 15000,
    'lstm_size': 100,
    'words': '../../../china_medical_char_data_cleaned/vocab.words.txt',
    'chars': '../../../china_medical_char_data_cleaned/vocab.chars.txt',
    'tags': '../../../china_medical_char_data_cleaned/vocab.tags.txt',
    'glove': '../../../medical_char_data_cleaned/glove.npz',
    'vector': 'bert_vec.npz'
}
model_path = './model/'


# parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
# parser.add_argument('--mode', type=str, default='train', help='train/test')
# args = parser.parse_args()

def train(sess, train, dev, epoch, gen, num_batches, batch, label):
    """
    :param train:
    :param dev:
    :return:
    """
    saver = tf.train.Saver(tf.global_variables())

    run_one_epoch(sess, train, dev, epoch, saver, gen, num_batches, batch, label)


def run_one_epoch(sess, words, labels, tags, dev, epoch, gen, num_batches, batch, label, it, iteration, saver):
    """
    :param sess:
    :param train:
    :param dev:
    :param tag2label:
    :param epoch:
    :param saver:
    :return:
    """
    #   num_batches = (len(train) + batch_size - 1) // batch_size

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if label == 0:
        seqs, labels = words, labels
        #        batches = batch_yield(train, batch_size, shuffle=True)
        #       for step, (seqs, labels) in batch:
        sys.stdout.write(' processing: epoch {} : {} batch / {} batches.'.format(epoch + 1, batch + 1, num_batches) + '\r')
        step_num = epoch * num_batches + batch + 1
        seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels)
        loss_train = gen.train(sess, seqs, seqs_len, labels, max_len)
        print(loss_train)
        print('11111111111111, training_phase_1 finished!')
    # elif label == 1:
    #     #        batches = batch_yield_for_unla_da(train, batch_size, shuffle=True)
    #     #        for step, (seqs, labels,tags) in enumerate(batches):
    #     seqs, labels, tags = words, labels, tags
    #     sys.stdout.write(' processing: {} batch / {} batches.'.format(batch + 1, num_batches) + '\r')
    #     step_num = epoch * num_batches + batch + 1
    #     seqs, seqs_len, labels, max_len = get_feed_dict_for_unlabel(seqs, labels)
    #     loss_train = gen.train_for_unlabel(sess, epoch, seqs, seqs_len, labels, tags, max_len, it, iteration, saver)
    #     print(loss_train)
    #     print('222222222222, training_ohase_II finished!')
    # elif label == 2:
    #
    #     seqs, labels, tags = words, labels, tags
    #
    #     sys.stdout.write(' processing: {} batch / {} batches.'.format(batch + 1, num_batches) + '\r')
    #     step_num = epoch * num_batches + batch + 1
    #     seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels)
    #     loss_train = gen.train_for_discri_labeled(sess, seqs, seqs_len, labels, tags, max_len)
    #     print(loss_train)
    #     print('333333333333333333333,labeled training of discriminator finised!')
    # else:
    #     seqs, labels, tags = words, labels, tags
    #     sys.stdout.write(' processing: {} batch / {} batches.'.format(batch + 1, num_batches) + '\r')
    #     step_num = epoch * num_batches + batch + 1
    #     seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels)
    #     loss_train = gen.train_for_discri_unlabeled(sess, epoch, seqs, seqs_len, labels, tags, max_len)
    #     print(loss_train)
    #     print('44444444444444444, unlabeled training of discriminator finised!')


def get_feed_dict(seqs, labels):
    seqs, seqs_len, max_len = pad_sequences(seqs, pad_mark='.')

    labels, _, _ = pad_sequences(labels, pad_mark='O')
    return seqs, seqs_len, labels, max_len


def get_feed_dict_for_unlabel(seqs, labels):
    seqs, seqs_len, max_len = pad_sequences(seqs, pad_mark='.')
    labels, _, _ = pad_sequences(labels, pad_mark='O')
    return seqs, seqs_len, labels, max_len


def get_metrics(sess, generator, dev, test_size, batch_size, flag=0):
    value_lis = []
    medi_lis = []
    metric_lis = generator.evaluate_ori(sess, dev, test_size, batch_size, flag=0)
    for ele in metric_lis:
        value_lis.append(ele.values())

    value_lis_transform = zip(*value_lis)
    for ele in value_lis_transform:
        transfor_ele = zip(*ele)
        for ele in transfor_ele:
            medi_lis.append(np.mean(ele))

    return medi_lis


def main():
    # if args.mode == 'train'
    ap = []
    with open('../../../china_medical_char_data_cleaned/vocab.tags.txt', 'r') as fin:
        for line in fin:
            ap.append(line.strip())
        fin.close()
    length = len(ap)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.625)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf.ConfigProto(
        #         device_count={ "CPU": 48 },
        #         inter_op_parallelism_threads=10,
        allow_soft_placement=True,
        #         intra_op_parallelism_threads=20,
        gpu_options=gpu_options))

    generator = Generator_BiLSTM_CRF(0.5, 1, batch_size, params, filter_sizes, num_filters, 0.75, length)
    generator.build_graph()

    tvars = tf.trainable_variables()
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
        tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    # 最后初始化变量
    # sess.run(tf.global_variables_initializer())

    sess.run(generator.init_op)
    sess.run(generator.table_op)
    sess.run(generator.init_op_1)
    saver = tf.train.Saver(tf.global_variables())

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        print("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    train_path = os.path.join('.', args.train_data, 'train_data1')
    train_unlabel_path = os.path.join('.', args.train_data_unlabel, 'train_unlabel')
    train_unlabel_path_1 = os.path.join('.', args.train_data_unlabel, 'train_unlabel1')
    test_path = os.path.join('.', args.test_data, 'test_data1')
    sub_test_path = os.path.join('.', args.sub_test_data, 'sub_test_data')
    train_data = read_corpus(train_path)
    train_data_unlabel = read_corpus_unlabel(train_unlabel_path)
    train_data_unlabel_1 = read_corpus_unlabel(train_unlabel_path_1)
    test_data = read_corpus(test_path);
    test_size = len(test_data)
    sub_test_data = read_corpus(sub_test_path)

    batches_labeled = batch_yield(train_data, batch_size, shuffle=True)
    batches_labeled = list(batches_labeled)
    # print(len(batches_labeled))
    num_batches = (len(train_data) + batch_size - 1) // batch_size
    batches_unlabeled = batch_yield_for_unla_da(train_data_unlabel, batch_size, shuffle=True)
    batches_unlabeled = list(batches_unlabeled)
    # print(len(batches_unlabeled))
    batches_labeled_for_dis = batch_yield_for_discri(train_data, batch_size, shuffle=True)
    batches_labeled_for_dis = list(batches_labeled_for_dis)
    batches_unlabeled_for_dis = batch_yield_for_discri_unlabeled(train_data_unlabel, batch_size, shuffle=True)
    batches_unlabeled_for_dis = list(batches_unlabeled_for_dis)
    dev = batch_yield(test_data, batch_size, shuffle=True)
    #    num_batches = min(len(batches_labeled),len(batches_unlabeled))
    num_batches_unlabel = (len(train_data_unlabel) + batch_size - 1) // batch_size
    num_batches_1 = min(len(batches_labeled_for_dis), len(batches_unlabeled_for_dis))
    index = 0
    if args.mode == 'train':
        for epoch_total in range(30):

            print('epoch_total and index are {} and {}'.format(epoch_total+1, index))
            medi_lis = get_metrics(sess, generator, dev, test_size, batch_size, flag=0)

            for ele in medi_lis:
                print('实体识别的', ele)
            print('the whole epoch training accuracy finished!!!!!!!!!!!!')

            for i, (words, labels) in enumerate(batches_labeled):
                run_one_epoch(sess, words, labels, tags=[], dev=test_data, epoch=epoch_total, gen=generator,
                              num_batches=num_batches, batch=i, label=0, it=0, iteration=0, saver=saver)

            dev1 = batch_yield(test_data, batch_size, shuffle=True)

            medi_lis_from_cross_entropy_training = get_metrics(sess, generator, dev1, test_size, batch_size, flag=0)

            for ele in medi_lis_from_cross_entropy_training:
                print('第一次', ele)

            print('the accuray after cross entropy training finished!!!!!!!!!!!!!!!!!!1')

            # if epoch_total > 3:
            #     #     batches_labeled_for_dis = batches_labeled_for_dis[0: len(batches_labeled_for_dis)-5]
            #     batch_dis_for_label = len(batches_labeled_for_dis)
            #     batch_dis_for_unlabel = len(batches_unlabeled_for_dis)
            #     for (ele, ele2) in zip(enumerate(batches_labeled_for_dis), enumerate(batches_unlabeled_for_dis)):
            #         index += 1
            #         #               if index > 70:
            #         #                    break
            #         run_one_epoch(sess, ele[1][0], ele[1][1], ele[1][2], dev=test_data, epoch=epoch_total,
            #                       gen=generator,
            #                       num_batches=batch_dis_for_label, batch=index, label=2, it=0, iteration=0, saver=saver)
            #         run_one_epoch(sess, ele2[1][0], ele2[1][1], ele2[1][2], dev=test_data, epoch=epoch_total,
            #                       gen=generator,
            #                       num_batches=batch_dis_for_unlabel, batch=index, label=3, it=0, iteration=0,
            #                       saver=saver)
            #     index = 0
            #
            #     print('the whole dis phaseI finished')
            #     #    index += 1
            #     for it in range(5):
            #         for i, (words, labels, tags) in enumerate(batches_unlabeled):
            #             #          print(i)
            #             run_one_epoch(sess, words, labels, tags=tags, dev=test_data, epoch=epoch_total, gen=generator,
            #                           num_batches=num_batches_unlabel, batch=i, label=1, it=it, iteration=i,
            #                           saver=saver)
            #
            #     dev2 = batch_yield(test_data, batch_size, shuffle=True)
            #
            #     medi_lis_from_adversarial_training = get_metrics(sess, generator, dev2, test_size, batch_size, flag=0)
            #
            #     for ele in medi_lis_from_adversarial_training:
            #         print('第二次打印', ele)
            #
            # print('the accuracy after adversarial training of generator finised!!!!!!!!!!!!!!')
            #
            # print('epoch {} finished!'.format(epoch_total))

    if args.mode == 'test':
        sub_dev = batch_yield_for_discri_unlabeled(sub_test_data, batch_size, shuffle=True)
        #          print(list(sub_dev))
        ckpt_file = tf.train.latest_checkpoint(model_path)

        generator = Generator_BiLSTM_CRF(0.5, batch_size, params, filter_sizes, num_filters, 0.75, length,
                                         is_training=False)
        generator.build_graph()
        generator.test(sess, sub_dev, test_size, 20)


if __name__ == '__main__':
    # if args.mode == 'train':
    main()


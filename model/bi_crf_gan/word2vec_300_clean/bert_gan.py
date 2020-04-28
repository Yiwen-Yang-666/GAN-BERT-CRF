import tensorflow as tf
import numpy as np
from pathlib import Path
from data import get_chunks, pad_sequences
from tf_metrics import precision, recall, f1
import os, time, csv
from bert_base.bert import modeling
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers
# 英文模型路径
# bert_path = '/home/ywd/tf_model/pre_training_model/uncased_L-12_H-768_A-12/'
# 中文模型路径
bert_path = '/home/ywd/tf_model/pre_training_model/chinese_L-12_H-768_A-12/'
init_checkpoint = os.path.join(bert_path, 'bert_model.ckpt')
bert_config_file  = os.path.join(bert_path, 'bert_config.json')
vocab_file = os.path.join(bert_path, 'vocab.txt')

bert_config = modeling.BertConfig.from_json_file(bert_config_file)

def bert_model(input_ids,is_training):
    with tf.Session() as sess:
        # input_ids = tf.placeholder(tf.int32, shape=[20, 128])
        # input_mask = tf.placeholder(tf.int32, shape=[20, 128])
        # token_type_ids= tf.placeholder(tf.int32, shape=[20, 128])
        input_mask = tf.zeros(shape=tf.shape(input_ids), dtype = np.int32)
        token_type_ids = tf.zeros(shape=tf.shape(input_ids), dtype = np.int32)
        model = modeling.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
                use_one_hot_embeddings=True
            )
        # 调用init_from_checkpoint方法
        # 最后初始化变量
        # graph = tf.get_default_graph()
        # tvars = tf.trainable_variables()
        #
        # (assignment_map,
        #  initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
        #     tvars, init_checkpoint)
        #
        # tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # # 初始化所有的变量
        #
        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)
        # sess.run(tf.global_variables_initializer())
        embeddings = model.get_sequence_output()
        return embeddings


class Generator_BiLSTM_CRF(object):
    def __init__(self, dropout, num_layers, batch_size, params, filter_sizes, num_filters, dropout_keep_Pro, length, is_training=True):
        #        self.dim = dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.batch_size = batch_size
        #        self.chars = chars
        #        self.tags = tags
        self.params = params
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_Pro
        self.stack_varia = tf.TensorArray(dtype=tf.float32, size=20, dynamic_size=True)
        self.stack_varia_1 = tf.TensorArray(dtype=tf.float32, size=20, dynamic_size=True)
        self.num_labels = length + 1 # num_labels
        self.is_training = is_training
        self.initializers = initializers

        with Path(self.params["tags"]).open() as f:
            self.indices = [idx for idx, tag in enumerate(f) ]#if tag.strip() != 'O']
            self.num_tags = len(self.indices) + 1
            print('value of indices is::', self.indices)
        self.vocab_words = tf.contrib.lookup.index_table_from_file(vocab_file,
                                                                   num_oov_buckets=self.params['num_oov_buckets'])
        self.vocab_tags = tf.contrib.lookup.index_table_from_file(self.params['tags'])
        # self.vector = np.load(self.params['vector'])['embeddings']  # np.array
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            # self.variable1 = np.vstack([self.vector, [[0.] * self.params['dim']]])
            # self.variable = tf.Variable(self.variable1, dtype=tf.float32, trainable=False)  #######4831 x 200
            self.variable_2 = tf.random_normal([self.num_labels, 768], stddev=0.1)
            self.variable_2 = tf.Variable(self.variable_2, dtype=tf.float32, trainable=False)

        self.hidden_unit = self.params['lstm_size']

    def build_graph(self):
        self.placeholder()

        self.build_bilstm_crf()
        self.discri_loss_op()
        self.discri_unlabeled_loss_op()
        self.trainstep_op()
        self.init_op()

    def placeholder(self):
        self.x = tf.placeholder(tf.string, shape=[None, None])
        self.sequence_length = tf.placeholder(tf.int32, shape=[None]) # batch下序列长度
        self.tags = tf.placeholder(tf.string, shape=[None, None]) # label
        self.y = tf.placeholder(tf.float32, shape=[None, 2])
        self.max_len = tf.placeholder(tf.int32)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()
        self.init_op_1 = tf.local_variables_initializer()
        self.table_op = tf.tables_initializer()

    def build_bilstm_crf(self, pred_ids=None):

        input_ids = self.vocab_words.lookup(self.x)
        # input_mask = tf.zeros(shape=tf.shape(input_ids), dtype = np.float32)
        # token_type_ids = tf.zeros(shape=tf.shape(input_ids), dtype = np.float32)
        # embeddings = tf.nn.embedding_lookup(self.variable, word_ids)  #### seq_length x batch_size x emb_dim
        # self.t = tf.transpose(embeddings, perm=[1, 0, 2])
        self.embedded_chars = bert_model(input_ids, is_training=self.is_training)

        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, 0.5)

        # lstm_output = self.blstm_layer(self.embedded_chars)
        # project
        # logits = self.project_bilstm_layer(lstm_output)
        # crf
        # loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        # self.pred_ids, self.best_score = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.sequence_length)

        # lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(self.params['lstm_size'])
        # lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(self.params['lstm_size'])
        # lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        #
        # output_fw, _ = lstm_cell_fw(self.embedded_chars , dtype=tf.float32, sequence_length=self.sequence_length)
        # output_bw, _ = lstm_cell_bw(self.embedded_chars , dtype=tf.float32, sequence_length=self.sequence_length)
        # output = tf.concat([output_fw, output_bw], axis=-1)
        # output = tf.transpose(output, perm=[1, 0, 2])  #######batch x sentence x emb（200）###这是bilist后的输出，即前向和后向的hidden_state相拼接
        # output = lstm_output
        output = self.embedded_chars
        ###########下面是CRF层，输出为self.pred_ids，即预测的id
        #
        logits = tf.layers.dense(output, self.num_tags)
        with tf.variable_scope('crf_param', reuse=tf.AUTO_REUSE):
            self.crf_params = tf.get_variable("crf", [self.num_tags, self.num_tags], dtype=tf.float32)


        self.weights = tf.sequence_mask(self.sequence_length)

        self.pred_ids, self.best_score = tf.contrib.crf.crf_decode(logits, self.crf_params, self.sequence_length)

        tags = self.vocab_tags.lookup(self.tags)
        self.tags_ids = tf.cast(tags, tf.int32)
        self.log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, tags, self.sequence_length, self.crf_params)
        self.loss1 = tf.reduce_mean(-self.log_likelihood)
        # self.loss1 = loss
        # total_loss, logits, trans, pred_ids
        dimen = 768
        vec_varia = tf.ones(shape=[dimen, 1], dtype=tf.int32)
        vec_varia_1 = tf.ones(shape=[dimen, 1], dtype=tf.int32)

        pred_ids = tf.nn.embedding_lookup(self.variable_2, self.pred_ids)
        tags = tf.nn.embedding_lookup(self.variable_2, tags)
        output_for_disc = output * pred_ids  #########batch x sentence x embedding
        output_for_disc_1 = output * tags
        samples = output_for_disc
        samples_1 = output_for_disc_1
        self.score = self.discriminator(samples)
        self.score_1 = self.discriminator(samples_1)
        ypred = tf.map_fn(lambda x: x[1], self.score)
        self.loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.score, labels=self.y)
        self.loss2 = tf.reduce_mean(self.loss2)
        m = tf.keras.metrics.Precision()
        m1 = tf.keras.metrics.Recall()
        m.update_state(self.tags_ids, self.pred_ids)
        m1.update_state(self.tags_ids, self.pred_ids)
        #            weights = tf.convert_to_tensor(weights)
        self.metrics = {
            'acc': tf.metrics.accuracy(self.tags_ids, self.pred_ids),# self.weights),
            'precision': precision(self.tags_ids, self.pred_ids, self.num_tags, self.indices, average='weighted'),# self.weights),
            'recall': recall(self.tags_ids, self.pred_ids, self.num_tags, self.indices, average='weighted'),# self.weights),
            'f1': f1(self.tags_ids, self.pred_ids, self.num_tags, self.indices, average='weighted'),# self.weights),
        }

    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        cell_tmp = rnn.BasicLSTMCell(self.hidden_unit)
        # 是否需要进行dropout
        if self.dropout is not None:
            cell_tmp = rnn.DropoutWrapper(cell_tmp, output_keep_prob=self.dropout)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.dropout is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout)
        return cell_fw, cell_bw

    def blstm_layer(self, embedding_chars):
        """

        :return:
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.max_len, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.tags_ids,
                transition_params=trans,
                sequence_lengths=self.sequence_length)
            return tf.reduce_mean(-log_likelihood), trans

    def linear(self, input_, output_size, scope=None):

        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]

        # Now the computation.
        with tf.variable_scope("discriminator_1", reuse=tf.AUTO_REUSE):
            matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
            bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

        return tf.matmul(input_, tf.transpose(matrix)) + bias_term

    def highway(self, input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """

        with tf.variable_scope(scope):
            for idx in range(num_layers):
                g = f(self.linear(input_, size, scope='highway_lin_%d' % idx))

                t = tf.sigmoid(self.linear(input_, size, scope='highway_gate_%d' % idx) + bias)

                output = t * g + (1. - t) * input_
                input_ = output

        return output

    def discriminator(self, x):
        l2_loss = tf.constant(0.0)
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            #         with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedded_chars = x
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            pooled_outputs = []
            for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, 768, 1, num_filter]
                    self.W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        self.W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.reduce_max(h, axis=1, keep_dims=True)
                    #        pooled = tf.nn.max_pool(
                    #            h,
                    #             ksize=[1, self.max_len - filter_size + 1, 1, 1],
                    #             strides=[1, 1, 1, 1],
                    #             padding='VALID',
                    #             name="pool")
                    pooled_outputs.append(pooled)

                    # Combine all the pooled features
            num_filters_total = sum(self.num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = self.highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

                # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

                # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, 2], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                ypred_for_auc = tf.nn.softmax(scores)
                #     self.predictions = tf.argmax(self.scores, 1, name="predictions")
                return scores

    def trainstep_op(self):
        #  with tf.variable_scope("train_step",reuse=tf.AUTO_REUSE):
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optim = tf.train.AdamOptimizer(0.0001)

        optim1 = tf.train.AdamOptimizer(1e-4)
        # 更新 bert 的 output层和crf
        print('=====================全部变量======================')
        print(tf.trainable_variables())
        self.params_0 = [param for param in tf.trainable_variables() if
                         ('crf_param' in param.name or 'output' in param.name )  and 'attention' not in param.name and 'discriminator_1' not in param.name and 'discriminator' not in param.name and 'embeddings' not in param.name ]
#        self.params_0 = [param for param in tf.trainable_variables() if
#                         ('crf_param' in param.name or 'output' in param.name) and 'discriminator' not in  param.name
#                          and 'discriminator_1' not in  param.name and 'attention' not in param.name
#                         and 'embeddings' not in param.name]
        print('====================loss1中更新的权重=======================')
        print(self.params_0)
        self.params_1 = [param for param in tf.trainable_variables() if
                         ('crf_param' not in param.name and 'bert/encoder/layer_11/output' in param.name) ]
        # 对抗生成器 更新bert的output层
#        self.params_1 = [param for param in tf.trainable_variables() if
#                         ('crf_param' not in param.name and 'bert/encoder/layer_11/output' in param.name)]
        print('=====================loss2中更新的权重======================')
        print(self.params_1)
        grads_and_vars = optim.compute_gradients(self.loss1, self.params_0)
        grads_and_vars2 = optim.compute_gradients(self.loss2, self.params_1)
        self.params = [param for param in tf.trainable_variables() if ('discriminator' in param.name)]
        print('======================loss3 and loss4更新的权重=====================')
        print(self.params)
        print('================================')
        grads_and_vars3 = optim1.compute_gradients(self.loss3, self.params)
        grads_and_vars4 = optim1.compute_gradients(self.loss4, self.params)

        grads_and_vars_clip = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars]
        grads_and_vars_clip2 = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars2]

        grads_and_vars_clip3 = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars3]
        grads_and_vars_clip4 = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars4]
        self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
        self.train_op2 = optim.apply_gradients(grads_and_vars_clip2, global_step=self.global_step)
        self.train_op3 = optim1.apply_gradients(grads_and_vars_clip3, global_step=self.global_step)
        self.train_op4 = optim1.apply_gradients(grads_and_vars_clip4, global_step=self.global_step)

    def evaluate_ori(self, sess, dev, test_size, batch_size, flag):
        #        iniit = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        metrc_lis = []
        index = 0
        for step, (seq, labels) in enumerate(dev):
            index += 1
            seqs, seqs_len, labels, _ = self.get_feed_dict(seq, labels)
            feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels}
            best_score, score, metrics, pred = sess.run([self.best_score, self.score, self.metrics, self.pred_ids],
                                                        feed_dict=feed)
            # print('真实结果',labels[1])
            # print('预测结果',pred[1])
            metrc_lis.append(metrics)

        return metrc_lis

    def evaluate(self, sess, dev, test_size, batch_size, flag):
        #        iniit = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        metrc_lis = []
        index = 0
        #    print('dev111111111111: ',list(dev))
        #     print('dev222222222222:', (list(dev))[-1])
        for step, (seq, labels, tags) in enumerate(dev):
            #   index += 1

            index += 1
            seqs, seqs_len, labels, _ = self.get_feed_dict(seq, labels)
            #       print(seqs[1])
            #       print(labels[1])

            #            for i in range(len(seqs)):
            #                print(str(i) + seqs[i])
            feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels, self.y: tags}
            best_score, score, loss, metrics = sess.run([self.best_score, self.score, self.loss2, self.metrics],
                                                        feed_dict=feed)
            print('score is', score)
            metrc_lis.append(metrics)

            if flag == 1:
                with open('./uncertainty_scheme/setence.txt', 'a') as fout:
                    #    besscore = []
                    for i in range(len(seqs)):
                        #            biglist = []
                        besscore = []
                        besscore.append(str(best_score[i]))
                        fout.writelines(p for p in seqs[i])
                        fout.write('\t')
                        fout.writelines(p for p in labels[i])
                        fout.write('\t')
                        fout.writelines(p for p in besscore)
                        fout.write('\t')
                        for p in list(score[i]):
                            #     print(i)
                            fout.write(str(p) + '\t')
                        #             fout.write('\t')
                        fout.write(str(loss))
                        fout.write('\n')
            #     fout.close()

            #  fout.writelines(p for p in list(score[i]))

        #                  fout.write(p for p in besscore)
        #                  fout.write(p for p in list(score[i]))
        #                  Ifout.write('\n')
        #            fout.write(seqs[i] + labels[i] + '\t' +besscore + '\t' + list(score[i]) + '\n')
        print('index is {}'.format(index))
        return metrc_lis

    #        return {"acc": 100 * accs, "f1": 100 * f1, "recall": 100 * r}

    def test(self, sess, dev, test_size, batch_size):

        #    var_name_list = [v.name for v in tf.trainable_variables()]
        #    for name in var_name_list:
        #        print(name)
        #       tf.reset_default_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            #       tf.reset_default_graph()
            saver.restore(sess, './model/1577729956-2')

            #        saver = tf.train.Saver()

            self.evaluate(sess, dev, test_size, batch_size, flag=1)

    def get_feed_dict(self, seqs, labels):
        seqs, seqs_len, max_len = pad_sequences(seqs, pad_mark='。')
        labels, _, _ = pad_sequences(labels, pad_mark='O')
        return seqs, seqs_len, labels, max_len

    def discri_loss_op(self):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.score_1, labels=self.y)
        self.loss3 = tf.reduce_mean(losses * 100)

    def discri_unlabeled_loss_op(self):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.score, labels=self.y)
        self.loss4 = tf.reduce_mean(losses * 100)

    def train(self, sess, seqs, seqs_len, labels, max_len):
        feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels, self.max_len: max_len}
        _, loss_train, step_num_ = sess.run([self.train_op, self.loss1, self.global_step], feed_dict=feed)
        #    print(pred[1])i
        return loss_train

    def train_for_unlabel(self, sess, epoch, seqs, seqs_len, labels, tags, max_len, it, iteration, saver):
        #        self.stack_varia = tf.TensorArray(dtype=tf.float32, size=20, dynamic_size=True)
        feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels, self.y: tags, self.max_len: max_len}
        _, loss_train, step_num_, W = sess.run([self.train_op2, self.loss2, self.global_step, self.W], feed_dict=feed)
        if it == 4 and iteration >= 79:
            # print(W)
            time_stamp = time.time()
            # 保存模型
            # self.model_path = os.path.join('./model', str(int(time_stamp)))
            # saver.save(sess, self.model_path, global_step=epoch)
        #    print('variable2 is oooooooooooooooooooooooooooo',vara2)
        return loss_train

    def train_for_discri_labeled(self, sess, seqs, seqs_len, labels, tags, max_len):
        feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels, self.y: tags, self.max_len: max_len}
        _, loss_disc_1, step_num = sess.run([self.train_op3, self.loss3, self.global_step], feed_dict=feed)
        return loss_disc_1

    def train_for_discri_unlabeled(self, sess, epoch, seqs, seqs_len, labels, tags, max_len):
        #  time_stamp = time.time()
        #    print(str(int(time_stamp)))
        #        self.model_path =os.path.join( './model',str(int(time_stamp)))
        feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels, self.y: tags, self.max_len: max_len}
        _, loss_disc_1, step_num = sess.run([self.train_op4, self.loss4, self.global_step], feed_dict=feed)
        #        saver.save(sess,self.model_path, global_step =epoch)
        #       print(W)
        return loss_disc_1

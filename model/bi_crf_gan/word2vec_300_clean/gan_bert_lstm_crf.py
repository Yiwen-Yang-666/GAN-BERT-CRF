import tensorflow as tf
import numpy as np
from pathlib import Path
# from main import get_feed_dict
from data import get_chunks, pad_sequences
from tf_metrics import precision, recall, f1
import os, time, csv


class Generator_BiLSTM_CRF(object):
    def __init__(self, dropout, params, filter_sizes, num_filters, num_classes, dropout_keep_Pro, length):
        #        self.dim = dim
        self.dropout = dropout
        #        self.batch_size = batch_size
        #        self.chars = chars
        #        self.tags = tags
        self.params = params
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_Pro
        self.stack_varia = tf.TensorArray(dtype=tf.float32, size=20, dynamic_size=True)
        self.stack_varia_1 = tf.TensorArray(dtype=tf.float32, size=20, dynamic_size=True)
        self.length = length

        with Path(self.params["tags"]).open() as f:
            self.indices = [idx for idx, tag in enumerate(f)]# if tag.strip() != 'O']
            self.num_tags = len(self.indices) + 1
            print('value of indices is::', self.indices)
        self.vocab_words = tf.contrib.lookup.index_table_from_file(self.params['words'],
                                                                   num_oov_buckets=self.params['num_oov_buckets'])
        self.vocab_tags = tf.contrib.lookup.index_table_from_file(self.params['tags'])
        self.vector = np.load(self.params['vector'])['embeddings']  # np.array
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self.variable1 = np.vstack([self.vector, [[0.] * self.params['dim']]])
            self.variable = tf.Variable(self.variable1, dtype=tf.float32, trainable=False)  #######4831 x 200
            self.variable_2 = tf.random_normal([self.length, 200], stddev=0.1)
            self.variable_2 = tf.Variable(self.variable_2, dtype=tf.float32, trainable=False)

    def build_graph(self):
        self.placeholder()

        self.build_bilstm_crf()
        #      self.discri_loss_op()
        self.discri_loss_op()
        self.discri_unlabeled_loss_op()

        self.trainstep_op()

        #       self.discri_loss_op()

        self.init_op()

    def placeholder(self):
        self.x = tf.placeholder(tf.string, shape=[None, None])
        self.sequence_length = tf.placeholder(tf.int32, shape=[None])
        self.tags = tf.placeholder(tf.string, shape=[None, None])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])
        self.max_len = tf.placeholder(tf.int32)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()
        self.init_op_1 = tf.local_variables_initializer()
        self.table_op = tf.tables_initializer()

    def build_bilstm_crf(self, pred_ids=None):

        word_ids = self.vocab_words.lookup(self.x)
        embeddings = tf.nn.embedding_lookup(self.variable, word_ids)  #### seq_length x batch_size x emb_dim
        self.t = tf.transpose(embeddings, perm=[1, 0, 2])
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(self.params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(self.params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        output_fw, _ = lstm_cell_fw(self.t, dtype=tf.float32, sequence_length=self.sequence_length)
        output_bw, _ = lstm_cell_bw(self.t, dtype=tf.float32, sequence_length=self.sequence_length)
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.transpose(output,
                              perm=[1, 0, 2])  #######batch x sentence x emb（200）###这是bilist后的输出，即前向和后向的hidden_state相拼接

        ###########下面是CRF层，输出为self.pred_ids，即预测的id

        logits = tf.layers.dense(output, self.num_tags)
        with tf.variable_scope('crf_param', reuse=tf.AUTO_REUSE):
            self.crf_params = tf.get_variable("crf", [self.num_tags, self.num_tags], dtype=tf.float32)
        self.weights = tf.sequence_mask(self.sequence_length)

        self.pred_ids, self.best_score = tf.contrib.crf.crf_decode(logits, self.crf_params, self.sequence_length)

        tags = self.vocab_tags.lookup(self.tags)
        self.tags_ids = tf.cast(tags, tf.int32)
        self.log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, tags, self.sequence_length, self.crf_params)
        self.loss1 = tf.reduce_mean(-self.log_likelihood)

        # total_loss, logits, trans, pred_ids
        dimen = 200
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
            'acc': tf.metrics.accuracy(self.tags_ids, self.pred_ids, self.weights),
            'precision': precision(self.tags_ids, self.pred_ids, self.num_tags, self.indices, self.weights),
            'recall': recall(self.tags_ids, self.pred_ids, self.num_tags, self.indices, self.weights),
            'f1': f1(self.tags_ids, self.pred_ids, self.num_tags, self.indices, self.weights),
        }

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
                    filter_shape = [filter_size, 200, 1, num_filter]
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
                W = tf.Variable(tf.truncated_normal([num_filters_total, self.num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
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

        self.params_1 = [param for param in tf.trainable_variables() if
                         'discriminator' not in param.name and 'crf_param' not in param.name]
        grads_and_vars = optim.compute_gradients(self.loss1)
        grads_and_vars2 = optim.compute_gradients(self.loss2, self.params_1)
        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
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
            print(pred[1])
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
                #           print('when flag euqal to 1')
                #           print(W)
                #           bess_core_l, scroe_l= [],[]
                #            for i in range(len(best_score)):
                #                    score.append(i)

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
            print(W)
            time_stamp = time.time()
            self.model_path = os.path.join('./model', str(int(time_stamp)))
            saver.save(sess, self.model_path, global_step=epoch)
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

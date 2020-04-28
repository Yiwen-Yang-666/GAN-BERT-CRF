import tensorflow as tf
import os
from bert_base.bert import modeling

bert_path = '/home/ywd/tf_model/pre_training_model/chinese_L-12_H-768_A-12/'
init_checkpoint = os.path.join(bert_path, 'bert_model.ckpt')
bert_config_file  = os.path.join(bert_path, 'bert_config.json')
vocab_file = os.path.join(bert_path, 'vocab.txt')

bert_config = modeling.BertConfig.from_json_file(bert_config_file)

with tf.Session() as sess:
    input_ids = tf.placeholder(tf.int32, shape=[20, 128])
    input_mask = tf.placeholder(tf.int32, shape=[20, 128])
    token_type_ids= tf.placeholder(tf.int32, shape=[20, 128])
    model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
            use_one_hot_embeddings=False
        )
    # 调用init_from_checkpoint方法
    # 最后初始化变量
    graph = tf.get_default_graph()
    tvars = tf.trainable_variables()

    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
        tvars, init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # 初始化所有的变量

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        print("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
    sess.run(tf.global_variables_initializer())
    embeddings = model.get_sequence_output()
    print(embeddings.shape)
import tensorflow as tf
from dataHelper import loadDataset, getBatches
from model import Seq2SeqModel
from tqdm import tqdm
import math
import os
import json
import time
import csv
from tensorflow.contrib.tensorboard.plugins import projector
import pickle
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


tf.app.flags.DEFINE_integer('rnn_size', 512, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 4, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 512, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 10, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 5, 'Maximum epochs of training')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 50, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_integer('train_size', 100, 'Training data size')
tf.app.flags.DEFINE_integer('validation_size', 50, 'Validation data')
tf.app.flags.DEFINE_boolean('add_noise', True, 'Add noise to sentences')
tf.app.flags.DEFINE_float('noise_prob', 0.1, 'omission and swap probability of denoise autoencoder')
tf.app.flags.DEFINE_boolean('use_attention', False, 'use attention or not')
tf.app.flags.DEFINE_boolean('use_pre_train', True, 'use gensim pre-train word embedding or not')
#要開始第二階段將下面三個lock, only_wuxia/only_normal, second_step改為True,若是要繼續第二階段則把second_step改false，其他一樣為True
tf.app.flags.DEFINE_boolean('lock', True, 'Training mode lock encoder')
tf.app.flags.DEFINE_boolean('only_wuxia', True, 'decoder only output wuxia words')
tf.app.flags.DEFINE_boolean('only_JINYONG', True, 'decoder only output wuxia words')
tf.app.flags.DEFINE_boolean('only_normal', False, 'decoder only output normal words')
tf.app.flags.DEFINE_boolean('second_step', True, 'second training of styleTransfer(use wuxia novel to train decoder)')
FLAGS = tf.app.flags.FLAGS

print(FLAGS)

# vocabulary路徑
# word2idFile = '../novel_sentence4/word2id'
# id2wordFile = '../novel_sentence4/id2word'

#金庸30萬跟白話30萬混合做出的vocab
word2idFile = 'data/step1_word2id'
id2wordFile = 'data/step1_id2word'

#若只要預測出武俠or白話單詞，將target_word2id改成對應風格的table
if FLAGS.only_wuxia:
    if FLAGS.only_JINYONG:
        target_word2idFile = 'data/decoder3_word2id_compress'
        target_id2wordFile = 'data/decoder3_id2word_compress'
    else:
        target_word2idFile = 'data/decoder_word2id_compress'
        target_id2wordFile = 'data/decoder_id2word_compress'
elif FLAGS.only_normal:
    target_word2idFile = 'data/decoder2_word2id_compress'
    target_id2wordFile = 'data/decoder2_id2word_compress'
else:
    target_word2idFile = word2idFile
    target_id2wordFile = id2wordFile 


#訓練資料路徑
if FLAGS.only_wuxia:
    if FLAGS.only_JINYONG:
        trainFile = 'data/JINYONG_train.txt'
        validationFile = 'data/JINYONG_validation.txt'
    else:
        trainFile = 'data/純武俠仙俠_seg5_clear_train.txt'
        validationFile = 'data/純武俠仙俠_seg5_clear_validation.txt'
elif FLAGS.only_normal:
    trainFile = 'data/非武俠_seg5_clear_train.txt'
    validationFile = 'data/非武俠_seg5_clear_validation.txt'
else:
    # trainFile = '../novel_sentence4/oneStep_sentence_clear_train.txt'
    # validationFile = '../novel_sentence4/oneStep_sentence_clear_validation.txt'
    trainFile = 'data/step1_train.txt'
    validationFile = 'data/step1_validation.txt'


#pre-train word embedding路徑
# pre_trainFile = '../novel_sentence4/gensim_5W.txt'
pre_trainFile = 'data/gensim_5W_2.txt'


print("word2idFile", word2idFile)
print("id2wordFile", id2wordFile)
print("trainFile", trainFile)
print("validationFile", validationFile)

word2id = json.load(open(word2idFile,'r'))
id2word = json.load(open(id2wordFile,'r'))
target_word2id = json.load(open(target_word2idFile,'r'))
target_id2word = json.load(open(target_id2wordFile,'r'))
pre_train_embedding = pickle.load(open(pre_trainFile,'rb')) #list type
trainingSamples = loadDataset(trainFile, word2idFile, FLAGS.train_size)
validationSamples = loadDataset(validationFile, word2idFile, FLAGS.validation_size)

#若沒限制第二階段只輸出武俠或只輸出白話，則output_vocab_size與input_vocab_size會相同
input_vocab_size = len(word2id)
output_vocab_size = len(target_word2id)


with tf.Session() as sess:

    #建立網路
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id, pre_train_embedding, input_vocab_size,
                         output_vocab_size, use_pre_train=FLAGS.use_pre_train, mode='train', use_attention=FLAGS.use_attention, beam_search=False,
                         beam_size=3, lock=FLAGS.lock, max_gradient_norm=5.0)

    print(np.shape(pre_train_embedding))

    global_params = tf.global_variables()

    #設定tf.train.Saver將目標只設定在encoder和global_step，之後讀取參數時只讀取這兩個
    if FLAGS.second_step: 
        variables_to_restore = [var for var in global_params if var.name.startswith('encoder') or var.name.startswith('global_step')]
        print("variables_to_restore")
        for i in variables_to_restore:
            print(i)
        model.saver =  tf.train.Saver(variables_to_restore, max_to_keep=10)

    #若model存在就載入參數，並將沒載入參數的部分初始化，若model不存在就將全部的參數初始化
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters...')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        if FLAGS.second_step: 
            variables_to_init = [var for var in global_params if not var.name.startswith('encoder') and not var.name.startswith('global_step')]
            print("variables_to_init")
            for i in variables_to_init:
                print(i)
            sess.run(tf.variables_initializer(variables_to_init))
    else:
        print('Created new model parameters...')
        sess.run(tf.global_variables_initializer())
        #若要使用pre-train word embedding這邊要run init_embedding()，將pre-train參數assign給embedding layer
        if FLAGS.use_pre_train:
            model.init_embedding(sess)


    #上面將save範圍給設定在encoder和global_step，必須改回來，以免之後儲存模型只儲存到部分網路
    model.saver =  tf.train.Saver(tf.global_variables(), max_to_keep=10)

    
    #設定tensorboard記錄檔的資料夾路徑，train和validation各一個
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir + '/train', graph=sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.model_dir + '/validation', graph=sess.graph)

    #tensorboard上的projector可視化
    config = projector.ProjectorConfig()
    embeddingConfig = config.embeddings.add()
    embeddingConfig.tensor_name = model.embedding.name
    # embeddingConfig.metadata_path = '/home/nj/styleTransfer/novel_sentence4/id2word.tsv'
    embeddingConfig.metadata_path = 'data/step1_id2word.tsv'
    projector.visualize_embeddings(summary_writer, config)


    #輸出網路參數，觀察變化
    variables_names =[v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k,v in zip(variables_names, values):
        if k == 'encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0':
            print(v.shape)
            print(k, v[0])
        if k == 'decoder_1/Attention_Wrapper/multi_rnn_cell/cell_1/lstm_cell/kernel:0':
            print(v.shape)
            print(k, v[0])


    validationbatches = getBatches(validationSamples, FLAGS.batch_size, FLAGS.noise_prob, id2word, target_word2id,
                                   FLAGS.only_wuxia, FLAGS.only_normal, FLAGS.add_noise)

    starttime = time.time()
    current_step = 0  
    #此迴圈控制Epochs
    for e in range(FLAGS.numEpochs):
        print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))
        batches = getBatches(trainingSamples, FLAGS.batch_size, FLAGS.noise_prob, id2word, target_word2id, FLAGS.only_wuxia, FLAGS.only_normal, FLAGS.add_noise)
        #此迴圈將一個Epoch所有的batches跑完
        for nextBatch in tqdm(batches, desc="Training"):
            loss, summary, global_step = model.train(sess, nextBatch)
            current_step += 1
            summary_writer.add_summary(summary, global_step)

            #每10個step輸出一次loss
            if current_step % 10 == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                tqdm.write("Step " + str(global_step) + " Minibatch Loss= " + "{:.6f}".format(loss))

            #每100個step跑一次validation
            if current_step % 100 == 0:
                try:
                    validationNextBatch = next(validationbatches)
                except:
                    validationbatches = getBatches(validationSamples, FLAGS.batch_size, FLAGS.noise_prob, id2word, target_word2id,
                                                   FLAGS.only_wuxia, FLAGS.only_normal, FLAGS.add_noise)
                    validationNextBatch = next(validationbatches)
                loss, summary= model.eval(sess, validationNextBatch)
                tqdm.write("Step " + str(global_step) + " Validation Loss= " + "{:.6f}".format(loss))
                validation_writer.add_summary(summary, global_step)

            #每FLAGS.steps_per_checkpoint個step跑一次儲存模型
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (global_step, loss, perplexity))
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=global_step)

    #跑完後印出部分參數，用於確認訓練結果
    values = sess.run(variables_names)
    for k,v in zip(variables_names, values):
        if k == 'encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0':
            print(v.shape)
            print(k, v[0])
        if k == 'decoder_1/Attention_Wrapper/multi_rnn_cell/cell_1/lstm_cell/kernel:0':
            print(v.shape)
            print(k, v[0])

    endtime = time.time()
    print("time: ",endtime-starttime)
            
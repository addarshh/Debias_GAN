
# coding: utf-8
# pretrain discriminator with real and fake tweets
# In[2]:

import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from restore_generator import RestoreGenerator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle as cPickle



# In[3]:

#########################################################################################
#  Generator  Hyper-parameters
# #####################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 10 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 600 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 512

#########################################################################################
#  Discriminator  Hyper-parameters
# ########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 512

#########################################################################################
#  Basic Training Parameters
# ########################################################################################
TOTAL_BATCH = 110
positive_file = 'data/tweets_seq_10.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000


# In[4]:

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

# In[5]:
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    vocab_size = 40620
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    PARAMS = cPickle.load(open('save/pretrain_gweights5.pickle'))


    generator = RestoreGenerator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,PARAMS)
    discriminator = Discriminator(sequence_length=10, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                    filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    # saver for discriminator only
    vars_to_restore=[v for v in tf.trainable_variables() if "discriminator" in v.name]
    vars_to_restore_dict = {}

    # make the dictionary, note that everything here will have “:0”, avoid it.
    for v in vars_to_restore:
        vars_to_restore_dict[v.name[:-2]] = v
    # saver
    saver = tf.train.Saver(vars_to_restore_dict)

    # init
    config = tf.ConfigProto(log_device_placement = True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())


    log = open('save/experiment-log-single-gpu-d.txt', 'w')

    print ('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for _ in range(1):
        generate_samples(sess, generator, BATCH_SIZE, generated_num*10, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file,10)
        for epoch in range(3):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _,d_loss = sess.run([discriminator.train_op,discriminator.loss], feed)

            print ('pre-train epoch ', epoch, 'loss ', d_loss)
            buffer = 'epoch:\t'+ str(it) + '\bce_d:\t' + str(d_loss) + '\n'
            log.write(buffer)


    log.close()

    # save weights
    saver.save(sess,"save/pretrain_d_model.ckpt")

if __name__ = '__main__':
    main()

""

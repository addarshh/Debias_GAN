
# coding: utf-8
# pretrain discriminator with real and fake tweets
# In[2]:

import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
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
PRE_EPOCH_NUM = 500 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 512

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


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    vocab_size = 40620
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    # load pretrained generator
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    config = tf.ConfigProto(log_device_placement = True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # supervised training
    gen_data_loader.create_batches(positive_file,10)

    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            print 'pre-train epoch ', epoch, 'loss ', loss
            buffer = 'epoch:\t'+ str(epoch) + '\tloss:\t' + str(loss) + '\n'
            log.write(buffer)

        if epoch % 100 == 0:
            # save weigths
            g_weights = sess.run([generator.g_embeddings,
                                 generator.Wi, generator.Ui, generator.bi,
                                 generator.Wf, generator.Uf, generator.bf,
                                 generator.Wog, generator.Uog, generator.bog,
                                 generator.Wc, generator.Uc, generator.bc,
                                 generator.Wo, generator.bo])
            with open('save/pretrain_g_weights_'+str(epoch)+'.pickle','w') as g_model:
                cPickle.dump(g_weights,g_model,protocol = cPickle.HIGHEST_PROTOCOL)


    # save final weights
    g_weights = sess.run([generator.g_embeddings,
                         generator.Wi, generator.Ui, generator.bi,
                         generator.Wf, generator.Uf, generator.bf,
                         generator.Wog, generator.Uog, generator.bog,
                         generator.Wc, generator.Uc, generator.bc,
                         generator.Wo, generator.bo

    with open('save/pretrain_g_weights.pickle','w') as g_model:
        cPickle.dump(g_weights,g_model,protocol = cPickle.HIGHEST_PROTOCOL)

    log.close()


if __name__ = '__main__':
    main()

""

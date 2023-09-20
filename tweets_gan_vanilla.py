
# the GAN without classifier structure
# the generator needs to converge before put the generator on GAN with classifier for debiasing

# coding: utf-8

# In[2]:

import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from restore_generator import RestoreGenerator
from discriminator import Discriminator
from rollout import ROLLOUT
import pickle as cPickle

""


#########################################################################################
#  Generator  Hyper-parameters
# #####################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 10 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64
g_learning_rate = 0.001
g_weigths = 'save/GAN1_g_weights_30.pickle' # pretrained g weigths to continue GAN training
#########################################################################################
#  Discriminator  Hyper-parameters
# ########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 512
d_weights = 'save/GAN1_d_weights_30.ckpt' # pretrained d wegiths to continue training
#########################################################################################
#  Basic Training Parameters
# ########################################################################################
GAN_STAGE_1 = 110
positive_file = 'data/tweets_seq_10.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000

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
    g_params = cPickle.load(open(g_weights))
    generator = RestoreGenerator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,g_params,g_learning_rate)

    discriminator = Discriminator(sequence_length=10, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto(log_device_placement = True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #sess.run(tf.global_variables_initializer())

    # restore discriminator
    # saver for discriminator only
    vars_to_restore=[v for v in tf.trainable_variables() if "discriminator" in v.name]
    vars_to_restore_dict = {}

    # make the dictionary
    for v in vars_to_restore:
        vars_to_restore_dict[v.name[:-2]] = v

    # saver restore last checkpoint
    saver = tf.train.Saver(vars_to_restore_dict)
    saver.restore(sess,d_weights)
    sess.run(tf.global_variables_initializer())

   # adversarial training
    rollout = ROLLOUT(generator, 0.8)
    log = open('save/experiment-log-gan-phase-1.txt', 'w')

    print ('#########################################################################')
    print ('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(GAN_STAGE_1):

        # supervised learning 5 epochs
        print ('Start supervised...')
        log.write('supervised training...\n')
        gen_data_loader.create_batches(positive_file,10)

        for epoch in xrange(0):
            loss = pre_train_epoch(sess, generator, gen_data_loader)
            print ('supervised epoch ', epoch, 'loss ', loss)
            buffer = 'epoch:\t'+ str(epoch+5*(total_batch)) + '\tsupervised_loss:\t' + str(loss) + '\n'
            log.write(buffer)

        print('total_batch:\t',total_batch,'supervised_loss:\t',loss)
        buffer = 'total_batch: ' + str(total_batch)+'tsupervised_loss: '+str(loss)+'\n'
        log.write(buffer)

        # Train the generator for 50 step
        for it in range(30):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 32, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _,g_loss = sess.run([generator.g_updates,generator.g_loss], feed_dict=feed)
            buffer = 'step:\t' + str(it+50*(total_batch)) + '\tg_loss:\t' + str(g_loss) + '\n'
            print ('step: ', it+50*(total_batch), 'g_loss: ', g_loss)
            log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Test
        buffer = 'total_batch:\t' + str(total_batch) + '\tg_loss:\t' + str(g_loss) + '\n'
        print ('total_batch: ', total_batch, 'g_loss: ', g_loss)
        log.write(buffer)

        # Train the discriminator
        for _ in range(1):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file,10)

            for _ in range(1):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _,d_loss = sess.run([discriminator.train_op,discriminator.loss], feed)
       # Test
        buffer = 'total_batch:\t' + str(total_batch) + '\td_loss:\t' + str(d_loss) + '\n'
        print ('total_batch: ', total_batch, 'd_loss: ', g_loss)
        log.write(buffer)

        # save weigths
        if total_batch % 30 == 0 or total_batch == GAN_STAGE_1-1:
            # g_weights
            g_weights = sess.run([generator.g_embeddings,
                                 generator.Wi, generator.Ui, generator.bi,
                                 generator.Wf, generator.Uf, generator.bf,
                                 generator.Wog, generator.Uog, generator.bog,
                                 generator.Wc, generator.Uc, generator.bc,
                                 generator.Wo, generator.bo])
            with open('save/GAN1_g_weights_'+str(total_batch)+'.pickle','w') as g_model:
                cPickle.dump(g_weights,g_model,protocol = cPickle.HIGHEST_PROTOCOL)


            # d_weights
            saver.save(sess,'save/GAN1_d_weights_'+str(total_batch)+'.ckpt')

    log.close()


if __name__ = '__main__':
    main()

""

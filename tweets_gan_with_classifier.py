'''
Define Hyper Parameters for generator, discriminator and classifier
Generate Samples for discriminator and classifier (race)
Process generated samples to original vocabulary, add padding
load pretrained generator and discriminator (Vanila version)
Make dictionary, load generated sequence, try with discriminator, try with classifier
Use roll-out with the classifier
Iterate over different training phases (Log and models gets stored)
'''

import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader,Cl_dataloader,CL_Data_Preproc
from generator import Generator
from restore_generator import RestoreGenerator
from discriminator_node_3 import Discriminator
from classifier import Classifier
from rollout_with_classifier import ROLLOUT_WITH_CLASSIFIER
import pickle as cPickle


#########################################################################################
#  Generator  Hyper-parameters
# #####################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 10 # sequence length
START_TOKEN = 0
SEED = 88
BATCH_SIZE = 64
g_learning_rate = 0.001

#########################################################################################
#  Basic Training Parameters
# ########################################################################################
GAN_STAGE_2 = 110
positive_file = 'data/tweets_seq_10.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000
vocab_size = 40620   # for GAN

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
#  Classifier  Hyper-parameters
# ########################################################################################
cl_embedding_dim = 64
cl_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] # Same structure of discriminator but bigger because sequence is lot longer
cl_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160] # Same structure of discriminator but bigger because sequence is lot longer
cl_dropout_keep_prob = 0.75
cl_l2_reg_lambda = 0.2
cl_batch_size = 512
cl_seq = 126
cl_vocab_size = 64000

# helper function to generate sampels for both discriminator and classifier
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
    dis_data_loader = Dis_dataloader(BATCH_SIZE)
    cl_data_loader = Cl_dataloader(BATCH_SIZE)

    # load pretrained generator and discriminator from pre-train, the vanila version.
    # load generator
    g_params = cPickle.load(open('save/GAN2_g_weights_small_batch_49.pickle'))
    generator = RestoreGenerator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,g_params,g_learning_rate)
    # load discriminator
    discriminator = Discriminator(sequence_length=10, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                    filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)
    # load classifier
    # Trained separetely, just for race on real data
    classifier = Classifier(sequence_length=cl_seq, num_classes=2, vocab_size=cl_vocab_size, embedding_size=cl_embedding_dim,
                                    filter_sizes=cl_filter_sizes, num_filters=cl_num_filters, l2_reg_lambda=cl_l2_reg_lambda)

    # setup session and intialize vars
    config = tf.ConfigProto(log_device_placement = True,allow_soft_placement = True)
    # config.gpu_options.visible_device_list = '2'
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    # restore discriminator
    # saver for discriminator only
    dis_vars_to_restore=[v for v in tf.trainable_variables() if "discriminator" in v.name]
    dis_vars_to_restore_dict = {}

    # make the dictionary
    for v in dis_vars_to_restore:
        dis_vars_to_restore_dict[v.name[:-2]] = v

    # saver restore last checkpoint
    dis_saver = tf.train.Saver(dis_vars_to_restore_dict)
    dis_saver.restore(sess,'save/GAN2_d_weights_small_batch_49.ckpt')

    # restore classifier
    # saver for classifier only
    cl_vars_to_restore=[v for v in tf.trainable_variables() if "classifier" in v.name]
    cl_vars_to_restore_dict = {}

    # make the dictionary
    for v in cl_vars_to_restore:
        cl_vars_to_restore_dict[v.name[:-2]] = v
    # saver
    cl_saver = tf.train.Saver(cl_vars_to_restore)
    cl_saver.restore(sess,'save/pretrain_cl_model.ckpt')

    # training phase 2
    # Roll out disciminator with classifier
    rollout = ROLLOUT_WITH_CLASSIFIER(generator, 0.8)

    log = open('save/experiment-log-gan-phase-2.txt', 'w')

    print ('#########################################################################')
    print ('Start Adversarial Training Phase 2...')
    log.write('adversarial training...\n')
    for total_batch in range(GAN_STAGE_2):

        # supervised learning 5 epochs
        print ('Start supervised...')
        log.write('supervised training...\n')
        gen_data_loader.create_batches(positive_file,10)

        supervised_g_loss = []
        for epoch in xrange(5):
            loss = pre_train_epoch(sess, generator, gen_data_loader)
            supervised_g_loss.append(loss)
            print ('supervised epoch ', epoch, 'loss ', np.mean(supervised_g_loss))
            buffer = 'epoch:\t'+ str(epoch+5*(total_batch)) + '\tave_supervised_loss:\t' + str(np.mean(supervised_g_loss)) + '\n'
            log.write(buffer)

        print('total_batch: ',total_batch,'supervised_loss: ',loss)
        buffer = 'total_batch:\t' + str(total_batch)+'\tsupervised_loss:\t'+str(loss)+'\n'
        log.write(buffer)

        # Train the generator for 10 step
        gan_g_loss = []
        for it in range(10):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 32, discriminator, classifier, c_weight=1)
            feed = {generator.x: samples, generator.rewards: rewards}
            _,g_loss = sess.run([generator.g_updates,generator.g_loss], feed_dict=feed)
            gan_g_loss.append(g_loss)
            buffer = 'step:\t' + str(it+10*(total_batch)) + '\tg_loss:\t' + str(g_loss) + '\n'
            print ('step: ', it+10*(total_batch), 'g_loss: ', g_loss)
            log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Test
        buffer = 'total_batch:\t' + str(total_batch) + '\tave_gan_g_loss:\t' + str(np.mean(gan_g_loss)) + '\n'
        print ('total_batch: ', total_batch, 'ave_g_loss: ', np.mean(gan_g_loss))
        log.write(buffer)

        # Train the discriminator
        for _ in range(1):
            generate_samples(sess, generator, BATCH_SIZE, generated_num*10, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file,10)
        # two epoches
            for _ in range(2):
                supervised_d_loss = []
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _,d_loss = sess.run([discriminator.train_op,discriminator.loss], feed)
                    supervised_d_loss.append(d_loss)
       # Test
        buffer = 'total_batch:\t' + str(total_batch) + '\td_loss:\t' + str(np.mean(supervised_d_loss)) + '\n'
        print ('total_batch: ', total_batch, 'd_loss: ', np.mean(supervised_d_loss))
        log.write(buffer)

        # save weigths
        if total_batch % 10 == 0 or total_batch=GAN_STAGE_2-1:
            # g_weights
            g_weights = sess.run([generator.g_embeddings,
                                 generator.Wi, generator.Ui, generator.bi,
                                 generator.Wf, generator.Uf, generator.bf,
                                 generator.Wog, generator.Uog, generator.bog,
                                 generator.Wc, generator.Uc, generator.bc,
                                 generator.Wo, generator.bo])
            with open('save/GAN2_g_weights_'+str(total_batch)+'.pickle','w') as g_model:
                cPickle.dump(g_weights,g_model,protocol = cPickle.HIGHEST_PROTOCOL)

            #d_weights
            dis_saver.save(sess,'save/GAN2_d_weights_'+str(total_batch)+'.ckpt')


    log.close()

if __name__ = '__main__':
    main()

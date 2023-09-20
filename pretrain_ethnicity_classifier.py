import numpy as np
import tensorflow as tf
import random
from dataloader import Cl_dataloader
from classifier import Classifier
import pickle as cPickle


#########################################################################################
#  Classifier  Hyper-parameters
# ########################################################################################
cl_embedding_dim = 64
cl_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
cl_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
cl_dropout_keep_prob = 0.75
cl_l2_reg_lambda = 0.2
cl_batch_size = 512
BATCH_SIZE = 512
cl_seq = 126

###############################################################################
#  Basic Training Parameters
# ########################################################################################
positive_file = 'data/cl_train_pos.txt'
negative_file = 'data/cl_train_neg.txt'
test_pos = 'data/cl_test_pos.txt'
test_neg = 'data/cl_test_neg.txt'


SEED = 88
START_TOKEN = 0
# In[5]:

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    vocab_size = 64000
    cl_data_loader = Cl_dataloader(BATCH_SIZE)

    classifier = Classifier(sequence_length=cl_seq, num_classes=2, vocab_size=vocab_size, embedding_size=cl_embedding_dim,
                                    filter_sizes=cl_filter_sizes, num_filters=cl_num_filters, l2_reg_lambda=cl_l2_reg_lambda)

    # saver for discriminator only
    vars_to_restore=[v for v in tf.trainable_variables() if "classifier" in v.name]
    vars_to_restore_dict = {}

    # make the dictionary
    for v in vars_to_restore:
        vars_to_restore_dict[v.name[:-2]] = v
    # saver
    saver = tf.train.Saver(vars_to_restore)

    config = tf.ConfigProto(log_device_placement = True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())


    log = open('save/experiment-log-single-gpu-classifier.txt', 'w')

    print ('Start training classifier...')
    cl_data_loader.load_train_data(positive_file, negative_file,cl_seq)
    for epoch in range(10):
        cl_data_loader.reset_pointer()
        for it in xrange(cl_data_loader.num_batch):
            x_batch, y_batch = cl_data_loader.next_batch()
            feed = {
                    classifier.input_x: x_batch,
                    classifier.input_y: y_batch,
                    classifier.dropout_keep_prob: cl_dropout_keep_prob
                }
            _,cl_loss = sess.run([classifier.train_op,classifier.loss], feed)

        print ('train epoch ', epoch, 'loss ', cl_loss)
        buffer = 'epoch:\t'+ str(it) + '\bce_classifier:\t' + str(cl_loss) + '\n'
        log.write(buffer)


    log.close()

    # save weights
    saver.save(sess,"save/pretrain_cl_model.ckpt")

if __name__ = '__main__':
    main()

""

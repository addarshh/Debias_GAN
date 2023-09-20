# baseline classifier for mention
# trained on all real tweets
# test performance and fairness metrics


import numpy as np
import tensorflow as tf
import random
from dataloader import Cl_dataloader
from mention_classifier_node_4 import MentionClassifier
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
positive_file = 'data/mention_train_pos.txt'
negative_file = 'data/mention_train_neg.txt'
test_pos = 'data/mention_test_pos.txt'
test_neg = 'data/mention_test_neg.txt'


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
    vars_to_restore=[v for v in tf.trainable_variables() if "mentionclassifier" in v.name]
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


    log = open('save/experiment-log-single-gpu-mention-classifier.txt', 'w')

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
    saver.save(sess,"save/pretrain_mentioncl_model.ckpt")


    # test performance
    all_score = []
    all_pred = []
    all_label = []
    all_seq = []
    cl_data_loader.load_train_data(test_pos, test_neg,cl_seq)
    for it in xrange(cl_data_loader.num_batch):
        x_batch, y_batch = cl_data_loader.next_batch()
        feed = {
                classifier.input_x: x_batch,
                classifier.dropout_keep_prob: cl_dropout_keep_prob
            }

        [score,y_pred] = sess.run([classifier.scores, classifier.predictions],feed)
        all_score.append(score)
        all_pred.append(y_pred)
        all_seq.append(np.array([seq_length(ele) for ele in x_batch]))
        all_label.append(np.array([item[1] for item in y_batch]))


    # In[102]:

    # flatten the list with list comprehension
    flat_label = [ele for sublist in all_label for ele in sublist]
    flat_pred = [ele for sublist in all_pred for ele in sublist]
    flat_seq = [ele for sublist in all_seq for ele in sublist]
    print(classification_report(flat_label,flat_pred))
    print(roc_auc_score(flat_label,flat_pred))


    # In[95]:

    # fairness
    # load labels
    with open('data/mention_test_neg_race_label.txt') as file_in:
        neg_race = []
        for line in file_in:
            neg_race.append(int(line.replace('\n','')))

    with open('data/mention_test_pos_race_label.txt') as file_in:
        pos_race = []
        for line in file_in:
            pos_race.append(int(line.replace('\n','')))

    # combine label
    race_label = np.array(pos_race+neg_race)[:len(flat_label)]


    # In[96]:

    # bias amplification for all and for 10 token seq
    original_pos_race = 1-np.mean(pos_race)
    predicted_pos_race = 1- float(sum([race_label[i]  if flat_pred[i]==1 else 0 for i in range(len(race_label))]))/float(sum(flat_pred))
    ba = (predicted_pos_race-original_pos_race)/original_pos_race
    print('true mention aa ratio: ', original_pos_race, 'predicted mention aa ratio: ', predicted_pos_race, 'amplification: ',ba)


    # In[97]:

    # doe fp
    fp_aa = float(sum([1-race_label[i]  if (flat_pred[i]==1 and flat_label[i]==0) else 0 for i in range(len(race_label))]))/float(len(race_label)-sum(race_label))
    fp_naa = float(sum([race_label[i]  if (flat_pred[i]==1 and flat_label[i]==0) else 0 for i in range(len(race_label))]))/float(sum(race_label))
    doe_fp = (fp_aa-fp_naa)/fp_naa
    print('fp aa ratio: ', fp_aa, 'fp naa ratio: ', fp_naa, 'doe: ',doe_fp)


    # In[98]:

    # doe fn
    fn_aa = float(sum([1-race_label[i]  if (flat_pred[i]==0 and flat_label[i]==1) else 0 for i in range(len(race_label))]))/float(len(race_label)-sum(race_label))
    fn_naa = float(sum([race_label[i]  if (flat_pred[i]==0 and flat_label[i]==1) else 0 for i in range(len(race_label))]))/float(sum(race_label))
    doe_fn = (fn_aa-fn_naa)/fn_naa
    print('fn aa ratio: ', fn_aa, 'fn naa ratio: ', fn_naa, 'doe: ',doe_fn)


    # In[109]:

    # for only seq = 10 tweets
    # test performance
    all_score = []
    all_pred = []
    all_label = []
    all_seq = []
    test_pos = 'data/mention_test_pos_10.txt'
    test_neg = 'data/mention_test_neg_10.txt'


    cl_data_loader.load_train_data(test_pos, test_neg,cl_seq)
    for it in xrange(cl_data_loader.num_batch):
        x_batch, y_batch = cl_data_loader.next_batch()
        feed = {
                classifier.input_x: x_batch,
                classifier.dropout_keep_prob: cl_dropout_keep_prob
            }

        [score,y_pred] = sess.run([classifier.scores, classifier.predictions],feed)
        all_score.append(score)
        all_pred.append(y_pred)
        all_seq.append(np.array([seq_length(ele) for ele in x_batch]))
        all_label.append(np.array([item[1] for item in y_batch]))

    # flatten the list with list comprehension
    flat_label = [ele for sublist in all_label for ele in sublist]
    flat_pred = [ele for sublist in all_pred for ele in sublist]
    flat_seq = [ele for sublist in all_seq for ele in sublist]
    print(classification_report(flat_label,flat_pred))
    print(roc_auc_score(flat_label,flat_pred))


    # In[111]:

    # fairness
    # fairness
    # load labels
    with open('data/mention_test_neg_race_label_10.txt') as file_in:
        neg_race = []
        for line in file_in:
            neg_race.append(int(line.replace('\n','')))

    with open('data/mention_test_pos_race_label_10.txt') as file_in:
        pos_race = []
        for line in file_in:
            pos_race.append(int(line.replace('\n','')))

    # combine label
    race_label = np.array(pos_race+neg_race)[:len(flat_label)]


    # In[113]:

    # bias amplification for all and for 10 token seq
    original_pos_race = 1-np.mean(pos_race)
    predicted_pos_race = 1- float(sum([race_label[i]  if flat_pred[i]==1 else 0 for i in range(len(race_label))]))/float(sum(flat_pred))
    ba = (predicted_pos_race-original_pos_race)/original_pos_race
    print('true mention aa ratio: ', original_pos_race, 'predicted mention aa ratio: ', predicted_pos_race, 'amplification: ',ba)

    # doe fp
    fp_aa = float(sum([1-race_label[i]  if (flat_pred[i]==1 and flat_label[i]==0) else 0 for i in range(len(race_label))]))/float(len(race_label)-sum(race_label))
    fp_naa = float(sum([race_label[i]  if (flat_pred[i]==1 and flat_label[i]==0) else 0 for i in range(len(race_label))]))/float(sum(race_label))
    doe_fp = (fp_aa-fp_naa)/fp_naa
    print('aa fp rate: ', fp_aa, 'naa fp rate: ', fp_naa, 'doe: ',doe_fp)

    # doe fn
    fn_aa = float(sum([1-race_label[i]  if (flat_pred[i]==0 and flat_label[i]==1) else 0 for i in range(len(race_label))]))/float(len(race_label)-sum(race_label))
    fn_naa = float(sum([race_label[i]  if (flat_pred[i]==0 and flat_label[i]==1) else 0 for i in range(len(race_label))]))/float(sum(race_label))
    doe_fn = (fn_aa-fn_naa)/fn_naa
    print('aa fn rate: ', fn_aa, 'naa fn rate: ', fn_naa, 'doe: ',doe_fn)


    # In[114]:

    # try it on synthetic data
    # for only seq = 10 tweets
    # test performance
    all_score = []
    all_pred = []
    all_label = []
    all_seq = []
    test_pos = 'data/mention_train_pos_gan.txt'
    test_neg = 'data/mention_train_neg_gan.txt'


    cl_data_loader.load_train_data(test_pos, test_neg,cl_seq)
    for it in xrange(cl_data_loader.num_batch):
        x_batch, y_batch = cl_data_loader.next_batch()
        feed = {
                classifier.input_x: x_batch,
                classifier.dropout_keep_prob: cl_dropout_keep_prob
            }

        [score,y_pred] = sess.run([classifier.scores, classifier.predictions],feed)
        all_score.append(score)
        all_pred.append(y_pred)
        all_seq.append(np.array([seq_length(ele) for ele in x_batch]))
        all_label.append(np.array([item[1] for item in y_batch]))

    # flatten the list with list comprehension
    flat_label = [ele for sublist in all_label for ele in sublist]
    flat_pred = [ele for sublist in all_pred for ele in sublist]
    flat_seq = [ele for sublist in all_seq for ele in sublist]
    print(classification_report(flat_label,flat_pred))
    print(roc_auc_score(flat_label,flat_pred))

if __name__ = '__main__':
    main()

""

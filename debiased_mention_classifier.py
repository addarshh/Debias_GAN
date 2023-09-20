'''
Train mention classifier with a mix of real and generated tweets with little ethnity clarity
To test performance, trained on mixed data and tested on real data
Determine classifier performance
Calculate the fairness using the fairness metrics
'''
import numpy as np
import tensorflow as tf
import random
from dataloader import Cl_dataloader
from mention_classifier_node_4 import MentionClassifier
import pickle as cPickle
# classification report
from sklearn.metrics import classification_report, roc_auc_score

#########################################################################################
#  Classifier  Hyper-parameters
# ########################################################################################
cl_embedding_dim = 64
cl_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
cl_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
cl_dropout_keep_prob = 0.75
cl_l2_reg_lambda = 0.2
cl_batch_size = 1024
BATCH_SIZE = 1024
cl_seq = 126
vocab_size = 64000

###############################################################################
#  Basic Training Parameters
# ########################################################################################
positive_file = 'data/mention_train_pos_mix_20.txt'
negative_file = 'data/mention_train_neg_mix_20.txt'
test_pos = 'data/mention_test_pos.txt'
test_neg = 'data/mention_test_neg.txt'
SEED = 88
START_TOKEN = 0

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    cl_data_loader = Cl_dataloader(BATCH_SIZE)

    classifier = MentionClassifier(sequence_length=cl_seq, num_classes=2, vocab_size=vocab_size, embedding_size=cl_embedding_dim,
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


    log = open('save/experiment-log-debaised_mention-classifier.txt', 'w')

    print ('Start training classifier...')
    cl_data_loader.load_train_data(positive_file, negative_file,cl_seq)
    for epoch in range(40):
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
        buffer = 'epoch:\t'+ str(epoch) + '\bce_classifier:\t' + str(cl_loss) + '\n'
        log.write(buffer)


    log.close()

    # save weights
    saver.save(sess,"save/train_mentioncl_model_with_mix_data_40.ckpt")


    # test performance
    test_pos = 'data/mention_test_pos.txt'
    test_neg = 'data/mention_test_neg.txt'
    all_score = []
    all_pred = []
    all_label = []
    cl_data_loader = Cl_dataloader(BATCH_SIZE,shuffle = False)

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

            all_label.append(np.array([item[1] for item in y_batch]))

    # flatten the list with list comprehension
    flat_label = [ele for sublist in all_label for ele in sublist]
    flat_pred = [ele for sublist in all_pred for ele in sublist]
    # Performance of classifier
    print(classification_report(flat_label,flat_pred))
    print(roc_auc_score(flat_label,flat_pred))


    # fairness
    # load the race labels
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

    ## Calculate Fairess Metrics
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


    # seq-10 (generate 10 sequence tokens)
    # for only seq = 10 tweets
    # test performance
    all_score = []
    all_pred = []
    all_label = []
    all_seq = []
    # Subset of test data with sequence 10 tokens
    test_pos = 'data/mention_test_pos_10.txt'
    test_neg = 'data/mention_test_neg_10.txt'

    cl_data_loader = Cl_dataloader(BATCH_SIZE,shuffle = False)

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

            all_label.append(np.array([item[1] for item in y_batch]))

    # flatten the list with list comprehension
    flat_label = [ele for sublist in all_label for ele in sublist]
    flat_pred = [ele for sublist in all_pred for ele in sublist]
    flat_seq = [ele for sublist in all_seq for ele in sublist]
    print(classification_report(flat_label,flat_pred))
    print(roc_auc_score(flat_label,flat_pred))

    # fairness for just 10 sequence tokens
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


if __name__ = '__main__':
    main()

""

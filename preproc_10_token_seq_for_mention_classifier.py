'''
load generator with particular set up and generate samples
process the sample tweets
convert to original vocabulary
pad to the the same max seq length
save as pos and neg train
use mixed data for training
classifier - @mention, generated data mixed with real data
'''

import numpy as np
import tensorflow as tf
import random
from dataloader import CL_Data_Preproc
from restore_generator import RestoreGenerator
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
pretrain_g_weights = 'save/GAN4_g_weights_small_batch_10.pickle'

#########################################################################################
#  Basic Training Parameters
# ########################################################################################
positive_file = 'data/mention_train_pos.txt'
negative_file = 'data/mention_train_neg.txt'
eval_file = 'save/eval_file.txt'
generated_num = 100000
mixing_factor = 30
vocab_size = 40620

#########################################################################################
#  revert to original vocab
# ########################################################################################
new_vocab_file = 'data/new_vocab.csv'
file_name = ['data/mention_train_pos_mix_30.txt','data/mention_train_neg_mix_30.txt']


# helper function to generate sampels for both descriminator and classifier
def generate_samples(sess, trainable_model, batch_size, generated_num):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    return(generated_samples)

# preproce the generated token to be the same corpus as the original
def preproc_for_classifier(generated_seq, cl_seq, new_vocab):
    cl_data_obj = CL_Data_Preproc()
    cl_data_obj.new_corpus(new_vocab)
    cl_data_obj.reverse_tweets(generated_seq)
    cl_data_obj.padding(cl_seq)
    return(cl_data_obj.result.astype('int32'))

# remove token 5, @mention token and pad to the same length
def remove_at_mention(tweet):
    tweet_int = [int(item) for item in tweet]
    tweet_int = list(filter(lambda x:x!=5,tweet_int))
    return(tweet_int)

def padding(data,max_seq):
    result = np.zeros([1,max_seq])
    result[:,:len(data)]=data
    return(result[0].astype(int))

def main():
    # load debiased generator
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    # Loading the generator
    g_params = cPickle.load(open(pretrain_g_weights))
    generator = RestoreGenerator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,g_params,g_learning_rate)

    # setup sssion and intialize vars
    config = tf.ConfigProto(log_device_placement = True,allow_soft_placement = True)
    # config.gpu_options.visible_device_list = '2'
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    # generate data (generated_num = No. of tweets generated)
    samples = generate_samples(sess,generator, batch_size=512, generated_num = generated_num*mixing_ratiro)

    # reverse to original vocab and max seq length
    processed_samples = preproc_for_classifier(samples,126,new_vocab = new_vocab_file)

    # remove 5 and pad again
    training_data = np.array([remove_at_mention(tweet) for tweet in processed_samples])
    training_data = [padding(ele,126) for ele in training_data]

    # find labels
    # if training_data is not the same as processed_samples, then the label is true
    label = np.array([0 if sum(abs(training_data[i]-processed_samples[i]))==0 else 1 for i in range(len(training_data))])


    # save the file as training
    pos_training = [training_data[i] for i in range(len(label))if label[i]==1]
    neg_training = [training_data[i] for i in range(len(label)) if label[i]!=1]

    # get real training data
    positive_examples = []
    with open(positive_file)as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            positive_examples.append(parse_line[:126])

    # negative examples
    negative_examples = []
    with open(negative_file)as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            negative_examples.append(parse_line[:126])

    # mix with real data
    pos = np.array(positive_examples + pos_training)
    neg = np.array(negative_examples + neg_training)

    # save to files
    files = [pos,neg]
    for i in range(len(files)):
        with open(file_name[i], 'w') as f:
            for item in files[i]:
                sentence = [ele for ele in item]
                sentence = str(sentence).replace(',','').replace('[','').replace(']','').replace('\n','')
                f.write("%s\n" % sentence)

if __name__ = '__main__':
    main()

""

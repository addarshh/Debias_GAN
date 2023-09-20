'''
Added 2 classes for the classifier
- Data processing, converting to original data corpus, padding
- load data, generate labels, shuffle data, split into bacthes
'''

import numpy as np
import csv

# Classifier data preporcessing, converted to original data corpus, pad it to required length of sequence
class CL_Data_Preproc():
    def __init__(self):
        pass

    # reverse lookup
    def new_corpus(self,new_vocab_file='data/new_vocab.csv'):
        corpus_val = []
        corpus_key = []
        with open(new_vocab_file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                corpus_val.append(int(row['original_idx']))
                corpus_key.append(int(row['index']))
        self.corpus_lookup = {corpus_key[i]: corpus_val[i] for i in range(len(corpus_key))}
    
    # reverse to the original tweets
    def reverse_tweets(self, generated_seq):
        # convert to original idx
        self.original_idx = []
        for ele in generated_seq:
            self.original_idx.append([int(self.corpus_lookup[int(idx)]) for idx in ele])
        self.original_idx = np.array(self.original_idx)

    def padding(self,max_seq):
        self.result = np.zeros([len(self.original_idx),max_seq])
        self.result[:self.original_idx.shape[0],:self.original_idx.shape[1]]=self.original_idx



class Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, data_file,seq_len):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == seq_len:
                    self.token_stream.append(parse_line[:seq_len])

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file,seq_len):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line[:seq_len])
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == seq_len:
                    negative_examples.append(parse_line[:seq_len])
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


# load data
class Cl_dataloader():
    def __init__(self, batch_size,shuffle = True):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.shuffle = shuffle


    def load_train_data(self, positive_file, negative_file,new_vocab=0,max_seq = 126):

        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line)<max_seq:
                    parse_line[len(parse_line):max_seq]=0
                positive_examples.append(parse_line)

        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line)<max_seq:
                    parse_line[len(parse_line):max_seq]=0
                negative_examples.append(parse_line)

        self.sentences = np.array(positive_examples + negative_examples)



        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        if self.shuffle:
            shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        else:
            shuffle_indices = np.arange(len(self.labels))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

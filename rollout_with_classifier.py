import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np
from dataloader import CL_Data_Preproc

class ROLLOUT_WITH_CLASSIFIER(object):
    def __init__(self, lstm, update_rate):
        self.lstm = lstm
        self.update_rate = update_rate

        self.num_emb = self.lstm.num_emb
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        self.start_token = tf.identity(self.lstm.start_token)
        self.learning_rate = self.lstm.learning_rate

        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

        #####################################################################################################
        # placeholder definition
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length]) # sequence of tokens generated by generator
        self.given_num = tf.placeholder(tf.int32)

        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))
        #####################################################################################################

        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        # When current index i < given_num, use the provided tokens as the input at each time step
        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            x_tp1 = ta_emb_x.read(i)
            gen_x = gen_x.write(i, ta_x.read(i))
            return i + 1, x_tp1, h_t, given_num, gen_x

        # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, given_num, gen_x

        i, x_t, h_tm1, given_num, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, self.given_num, gen_x))

        _, _, _, _, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, self.gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length


    def get_reward(self, sess, input_x, rollout_num, discriminator, classifier, c_weight=1, max_seq_for_classifier=126,new_vocab_file = 'data/new_vocab.csv'):

        d_rewards = [] # discriminator reward
        c_rewards = [] # classifier reward
        cl_preproc = CL_Data_Preproc() # preprocess the data from generator directly
        cl_preproc.new_corpus(new_vocab_file)

        for i in range(rollout_num):
            # given_num between 1 to sequence_length - 1 for a part completed sentence
            for given_num in range(1, self.sequence_length ):
                feed = {self.x: input_x, self.given_num: given_num}
                samples = sess.run(self.gen_x, feed)

                # discriminator reward
                feed = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    d_rewards.append(ypred)
                else:
                    d_rewards[given_num - 1] += ypred

                # classifier reward
                # preprocess data
                cl_preproc.reverse_tweets(samples)
                cl_preproc.padding(max_seq_for_classifier)
                classifier_sample = cl_preproc.result.astype('int32')
                feed = {classifier.input_x: classifier_sample, classifier.dropout_keep_prob: 1.0}
                ypred_for_auc,y_pred = sess.run([classifier.ypred_for_auc,classifier.predictions], feed)
                yprob = np.array([item[1] for item in ypred_for_auc])
                # remove extreme values so we do not end up with nan or inf
                yprob_filter = np.where(yprob ==0, 1e-4, yprob)
                sub_yprob_filter = np.where(yprob_filter == 1, 1e-4, 1-yprob_filter)
                # reverse bce reward
                bce = -(np.log(sub_yprob_filter)*(1-ypred)+np.log(yprob_filter)*(ypred))
                penalize_one = 1-yprob
                if i == 0:
                    # binary cross entropy reverse loss (maxmize this)
                    #c_rewards.append(bce)
                    # penalize 1 class
                    c_rewards.append(penalize_one)
                else:
                    # bce reverse loss
                    #c_rewards[given_num - 1] += bce
                    # penalize 1 class
                    c_rewards[given_num - 1] += penalize_one



            # the last token reward
            # discriminator reward
            feed = {discriminator.input_x: input_x, discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                d_rewards.append(ypred)
            else:
                # completed sentence reward
                d_rewards[self.sequence_length - 1] += ypred

            # classifier reward
            cl_preproc.reverse_tweets(input_x)
            cl_preproc.padding(max_seq_for_classifier)
            classifier_sample = cl_preproc.result.astype('int32')
            feed = {classifier.input_x: classifier_sample, classifier.dropout_keep_prob: 1.0}
            ypred_for_auc,y_pred = sess.run([classifier.ypred_for_auc, classifier.predictions],feed)
            yprob = np.array([item[1] for item in ypred_for_auc])
            # remove extreme values so we do not end up with nan or inf
            yprob_filter = np.where(yprob ==0, 1e-4, yprob)
            sub_yprob_filter = np.where(yprob_filter ==1, 1e-4, 1-yprob_filter)
            # reverse bce reward
            bce = -(np.log(sub_yprob_filter)*(1-ypred)+np.log(yprob_filter)*(ypred))
            penalize_one = 1-yprob

            # Penalize 0 and 1 reward, 
            if i == 0:
                #c_rewards.append(bce) # binary crossentropy version
                # penalize 1 class
                c_rewards.append(penalize_one)
            else:
                # completed sentence reward
                #c_rewards[self.sequence_length - 1] += bce
                # penalize 1 class
                c_rewards[self.sequence_length -1] +=penalize_one

        # discriminator reward
        d_rewards = np.transpose(np.array(d_rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        # classifier reward
        c_rewards = np.transpose(np.array(c_rewards)) / (1.0 * rollout_num)

        # average it by hyperparameter lambda
        rewards =  d_rewards + c_weight*c_rewards # c_weight - lambda value

        return rewards

    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.identity(self.lstm.Wi)
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wog = tf.identity(self.lstm.Wog)
        self.Uog = tf.identity(self.lstm.Uog)
        self.bog = tf.identity(self.lstm.bog)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def update_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = self.update_rate * self.Wi + (1 - self.update_rate) * tf.identity(self.lstm.Wi)
        self.Ui = self.update_rate * self.Ui + (1 - self.update_rate) * tf.identity(self.lstm.Ui)
        self.bi = self.update_rate * self.bi + (1 - self.update_rate) * tf.identity(self.lstm.bi)

        self.Wf = self.update_rate * self.Wf + (1 - self.update_rate) * tf.identity(self.lstm.Wf)
        self.Uf = self.update_rate * self.Uf + (1 - self.update_rate) * tf.identity(self.lstm.Uf)
        self.bf = self.update_rate * self.bf + (1 - self.update_rate) * tf.identity(self.lstm.bf)

        self.Wog = self.update_rate * self.Wog + (1 - self.update_rate) * tf.identity(self.lstm.Wog)
        self.Uog = self.update_rate * self.Uog + (1 - self.update_rate) * tf.identity(self.lstm.Uog)
        self.bog = self.update_rate * self.bog + (1 - self.update_rate) * tf.identity(self.lstm.bog)

        self.Wc = self.update_rate * self.Wc + (1 - self.update_rate) * tf.identity(self.lstm.Wc)
        self.Uc = self.update_rate * self.Uc + (1 - self.update_rate) * tf.identity(self.lstm.Uc)
        self.bc = self.update_rate * self.bc + (1 - self.update_rate) * tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self):
        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_output_unit(self):
        self.Wo = self.update_rate * self.Wo + (1 - self.update_rate) * tf.identity(self.lstm.Wo)
        self.bo = self.update_rate * self.bo + (1 - self.update_rate) * tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_params(self):
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.update_recurrent_unit()
        self.g_output_unit = self.update_output_unit()


def generated_sample_classify_preproc(sample,max_seq):
    sentences = np.array([])
    result = np.zeros([len(sample),max_seq])
    result[:sample.shape[0],:sample.shape[1]]=sample
    return(result)

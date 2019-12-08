import os
#os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
import numpy as np
import cPickle as pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from scipy import io
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
from keras.models import load_model
np.random.seed(1234)

r = robjects.r
rpy2.robjects.numpy2ri.activate()

#np.set_printoptions(threshold = 1e6)
importr('genlasso')
importr('gsubfn')

def perf_measure(y_true, y_pred):
    TP_FN = np.count_nonzero(y_true)
    FP_TN = y_true.shape[0] * y_true.shape[1] - TP_FN
    FN = np.where((y_true - y_pred) == 1)[0].shape[0]
    TP = TP_FN - FN
    FP = np.count_nonzero(y_pred) - TP
    TN = FP_TN - FP
    Precision = float(float(TP) / float(TP + FP + 1e-9))
    Recall = float(float(TP) / float((TP + FN + 1e-9)))
    accuracy = float(float(TP + TN) / float((TP_FN + FP_TN + 1e-9)))
    F1 =  2*((Precision * Recall) / (Precision + Recall))
    return Precision, Recall, accuracy

class xai_rnn(object):
    """class for explaining the rnn prediction"""
    def __init__(self, model, data, start_binary, real_start_sp):
        """
        Args:
            model: target rnn model.
            data: data sample needed to be explained.
            start: value of function start.
        """
        self.model = model
        self.data = data
        self.seq_len = data.shape[1]
        self.start = start_binary
        self.sp = np.where((self.data == self.start))
        self.real_sp = real_start_sp
        self.pred = self.model.predict(self.data, verbose = 0)[self.sp]

    def truncate_seq(self, trunc_len):
        """ Generate truncated data sample
        Args:
            trun_len: the lenght of the truncated data sample.

        return:
            trunc_data: the truncated data samples.
        """
        self.trunc_data_test = np.zeros((1, self.seq_len), dtype=int)
        #self.trunc_data_test = np.ones((1, self.seq_len),dtype = int)
        self.tl = trunc_len
        cen = self.seq_len/2
        half_tl = trunc_len/2

        if self.real_sp < half_tl:
            self.trunc_data_test[0, (cen - self.real_sp):cen] = self.data[0, 0:self.real_sp]
            self.trunc_data_test[0, cen:(cen+half_tl+1)] = self.data[0, self.real_sp:(self.real_sp+half_tl+1)]

        elif self.real_sp >= self.seq_len - half_tl:
            self.trunc_data_test[0, (cen - half_tl):cen] = self.data[0, (self.real_sp-half_tl):self.real_sp]
            self.trunc_data_test[0, cen:(cen + (self.seq_len-self.real_sp))] = self.data[0, self.real_sp:self.seq_len]

        else:
            self.trunc_data_test[0, (cen - half_tl):(cen + half_tl + 1)] = self.data[0, (self.real_sp - half_tl):(self.real_sp + half_tl + 1)]

        self.trunc_data = self.trunc_data_test[0, (cen - half_tl):(cen + half_tl + 1)]
        return self.trunc_data

    def xai_feature(self, samp_num, option= 'None'):
        """extract the important features from the input data
        Arg:
            fea_num: number of features that needed by the user
            samp_num: number of data used for explanation
        return:
            fea: extracted features
        """
        cen = self.seq_len/2
        half_tl = self.tl/2
        sample = np.random.randint(1, self.tl+1, samp_num)
        features_range = range(self.tl+1)
        data_explain = np.copy(self.trunc_data).reshape(1, self.trunc_data.shape[0])
        data_sampled = np.copy(self.data)
        for i, size in enumerate(sample, start=1):
            inactive = np.random.choice(features_range, size, replace=False)
            tmp_sampled = np.copy(self.trunc_data)
            tmp_sampled[inactive] = 0
            tmp_sampled = tmp_sampled.reshape(1, self.trunc_data.shape[0])
            data_explain = np.concatenate((data_explain, tmp_sampled), axis=0)
            data_sampled_mutate = np.copy(self.data)
            if self.real_sp < half_tl:
                data_sampled_mutate[0, 0:tmp_sampled.shape[1]] = tmp_sampled
            elif self.real_sp >= self.seq_len - half_tl:
                data_sampled_mutate[0, (self.seq_len - tmp_sampled.shape[1]): self.seq_len] = tmp_sampled
            else:
                data_sampled_mutate[0, (self.real_sp - half_tl):(self.real_sp + half_tl + 1)] = tmp_sampled
            data_sampled = np.concatenate((data_sampled, data_sampled_mutate),axis=0)

        if option == "Fixed":
            print "Fix start points"
            data_sampled[:, self.real_sp] = self.start
        label_sampled = self.model.predict(data_sampled, verbose = 0)[:, self.real_sp, 1]
        label_sampled = label_sampled.reshape(label_sampled.shape[0], 1)
        X = r.matrix(data_explain, nrow = data_explain.shape[0], ncol = data_explain.shape[1])
        Y = r.matrix(label_sampled, nrow = label_sampled.shape[0], ncol = label_sampled.shape[1])

        n = r.nrow(X)
        p = r.ncol(X)
        results = r.fusedlasso1d(y=Y,X=X)
        result = np.array(r.coef(results, np.sqrt(n*np.log(p)))[0])[:,-1]

        importance_score = np.argsort(result)[::-1]
        self.fea = (importance_score-self.tl/2)+self.real_sp
        self.fea = self.fea[np.where(self.fea<200)]
        self.fea = self.fea[np.where(self.fea>=0)]
        return self.fea

class fid_test(object):
    def __init__(self, xai_rnn):
        self.xai_rnn = xai_rnn

    def pos_boostrap_exp(self, num_fea):
        """
        feature deduction test.
        :param num_fea: number of selected features.
        :return: generated testing sample, probability of our method and random selection.
        """
        test_data = np.copy(self.xai_rnn.data)
        selected_fea = self.xai_rnn.fea[0:num_fea]
        test_data[0, selected_fea] = 0
        P1 = self.xai_rnn.model.predict(test_data, verbose=0)[0, self.xai_rnn.real_sp,1]
        #test_data[0, self.xai_rnn.sp] = self.xai_rnn.start
        #P2 = self.xai_rnn.model.predict(test_data, verbose=0)[0, self.xai_rnn.real_sp,1]

        random_fea = np.random.randint(0, 200, num_fea)
        test_data_1 = np.copy(self.xai_rnn.data)
        test_data_1[0, random_fea] = 0
        P2 = self.xai_rnn.model.predict(test_data_1, verbose=0)[0, self.xai_rnn.real_sp, 1]
        return test_data, P1, P2


    def neg_boostrap_exp(self, test_seed, num_fea):
        """
        feature augmentation test.
        :param num_fea: number of selected features.
        :return: generated testing sample, probability of our method and random selection.
        """
        test_seed = test_seed.reshape(1, 200)
        test_data = np.copy(test_seed)
        selected_fea = self.xai_rnn.fea[0:num_fea]
        test_data[0, selected_fea] = self.xai_rnn.data[0,selected_fea]
        P_neg_1 = self.xai_rnn.model.predict(test_data, verbose=0)[0, self.xai_rnn.real_sp,1]

        random_fea = np.random.randint(0, 200, num_fea)
        test_data_1 = np.copy(test_seed)
        test_data_1[0, random_fea] = self.xai_rnn.data[0,random_fea]
        P_neg_2 = self.xai_rnn.model.predict(test_data_1, verbose=0)[0, self.xai_rnn.real_sp, 1]

        return test_data, P_neg_1, P_neg_2


    def new_test_exp(self, num_fea):
        """
        Synthetic test.
        :param num_fea: number of selected features.
        :return: generated testing sample, probability of our method and random selection.
        """
        test_data = np.zeros_like(self.xai_rnn.data)
        # test_data = np.ones_like(self.xai_rnn.data) # either use one or zero to pad the missing part.
        selected_fea = self.xai_rnn.fea[0:num_fea]
        test_data[0, selected_fea] = self.xai_rnn.data[0, selected_fea]
        P_test_1 = self.xai_rnn.model.predict(test_data, verbose=0)[0, self.xai_rnn.real_sp,1]

        random_fea = np.random.randint(0, 200, num_fea)
        test_data_1 = np.zeros_like(self.xai_rnn.data)
        test_data_1[0, random_fea] = self.xai_rnn.data[0, random_fea]
        P_test_2 = self.xai_rnn.model.predict(test_data_1, verbose=0)[0, self.xai_rnn.real_sp, 1]

        return test_data, P_test_1, P_test_2


if __name__ == "__main__":
    print '[Load model...]'
    model = load_model('../model/O1_Bi_Rnn.h5')
    PATH_TEST_DATA = '../data/elf_x86_32_gcc_O1_test.pkl'
    n_fea_select = 25

    print '[Load data...]'
    data = pickle.load(file(PATH_TEST_DATA))
    data_num = len(data[0])
    print 'Data_num:',data_num
    seq_len = len(data[0][0])
    print 'Sequence length:', seq_len

    # Padding sequence and prepare labelss
    x_test = pad_sequences(data[0], maxlen=seq_len, dtype='int32', padding='post', truncating='post', value=0)
    x_test = x_test + 1
    y = pad_sequences(data[1], maxlen=seq_len, dtype='int32', padding='post', truncating='post', value=0)
    y_test = np.zeros((data_num, seq_len, 2), dtype=y.dtype)
    for test_id in xrange(data_num):
        y_test[test_id, np.arange(seq_len), y[test_id]] = 1

    # extract all the function starts for explanation
    idx = np.nonzero(y)[0]
    start_points = np.nonzero(y)[1]

    n1 = idx.shape[0]
    print n1
    n2 = start_points.shape[0]
    print n2

    n_pos = 0
    n_new = 0
    n_neg = 0

    n_pos_rand = 0
    n_new_rand = 0
    n_neg_rand = 0
    n = 0

    for i in xrange(len(x_test)):
        if i % 200 == 0:
            print('%d of %d' % (i, len(x_test)))
        if i in idx:
            idx_Col = np.where(idx == i)
            idx_Row = start_points[idx_Col]
            binary_func_start = x_test[i][idx_Row]
            x_test_d = x_test[i:i + 1]

            for j in xrange(len(idx_Row)):
                xai_test = xai_rnn(model, x_test_d, binary_func_start[j], idx_Row[j])
                if xai_test.pred[0, 1] > 0.5:
                    n = n + 1
                    truncate_seq_data = xai_test.truncate_seq(40)
                    xai_fea = xai_test.xai_feature(500)
                    fea = np.zeros_like(xai_test.data)
                    fea[0, xai_fea[0:25]] = xai_test.data[0, xai_fea[0:25]]
                    fid_tt = fid_test(xai_test)

                    test_data, P1, P2 = fid_tt.pos_boostrap_exp(n_fea_select)
                    if P1 > 0.5:
                       n_pos = n_pos + 1
                    if P2 > 0.5:
                       n_pos_rand = n_pos_rand + 1

                    test_data, P_test_1, P_test_2 = fid_tt.new_test_exp(n_fea_select)
                    if P_test_1> 0.5:
                       n_new = n_new + 1
                    if P_test_2 > 0.5:
                       n_new_rand = n_new_rand + 1

                    test_seed = x_test[0, ]
                    neg_test_data, P_neg_1, P_neg_2 = fid_tt.neg_boostrap_exp(test_seed, n_fea_select)
                    if P_neg_1 > 0.5:
                       n_neg = n_neg + 1
                    if P_neg_2 > 0.5:
                       n_neg_rand = n_neg_rand + 1

    print 'Our method'
    print 'Acc pos:', float(n_pos)/n
    print 'Acc new:', float(n_new)/n
    print 'Acc neg:', float(n_neg)/n

    print 'Random'
    print 'Acc pos:', float(n_pos_rand) / n
    print 'Acc new:', float(n_new_rand) / n
    print 'Acc neg:', float(n_neg_rand) / n
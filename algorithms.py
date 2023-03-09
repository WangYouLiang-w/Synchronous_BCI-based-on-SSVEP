#%% eTRCA
import numpy as np
import numpy.matlib
import math
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from SKLDA import SKLDA
from draw_figure import draw_fig_SVM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def corr2(a, b):
    a = a - np.sum(a) / np.size(a)
    b = b - np.sum(b) / np.size(b)
    r = (a * b).sum() / math.sqrt((a*a).sum()*(b*b).sum())
    return r

'''
》1、信号的特征提取和分类（eTRCA、R_BL)
》2、状态检测的特征提取方法(getFuture)
》3、分类方法（LDA、SVM）


'''
class eTRCA:
    def __init__(self) -> None:
        pass


    def trca_matrix(self, data):
        """
        Task-related component analysis (TRCA)
        input:
        X : eeg data (Num of channels * num of sample points * number of trials)
        
        output:
        w: spatial filter
        """
        X = data
        nchans = X.shape[0]
        nTimes = X.shape[1]
        nTrial = X.shape[2]

        # solves s
        S = np.zeros((nchans, nchans))

        for i in range(nTrial):
            for j in range(nTrial):
                if (i != j):
                    x_i = X[:, :, i]
                    x_j = X[:, :, j]
                    S = S + np.dot(x_i, (x_j.T))
        
        # solves Q
        X1 = X.reshape([nchans, nTimes * nTrial], order = 'F')
        a = numpy.matlib.repmat(np.mean(X1,1), X1.shape[1], 1)
        X1 = X1 - a.T
        Q = X1 @ X1.T
        # get eigenvector
        b = np.dot(np.linalg.inv(Q), S)
        [eig_value, eig_w] = np.linalg.eig(b)

        # Descending order
        eig_w = eig_w[:, eig_value.argsort()[::-1]] # return indices in ascending order and reverse
        eig_value.sort()
        eig_value = eig_value[::-1] # sort in descending

        w = eig_w[:,0:2]
        return w.real


    def TRCA_train(self, trainData):
        """
        input:
        train_data  (Num of channels * num of sample points * num of events * number of train trials)
        
        output:
        w （Num of channels * Num of events）
        mean_temp （Num of channels *Num of sample points * Num of events）
        """
        [nChannels, nTimes, nEvents, nTrials] = trainData.shape
        # get w of event class
        w = np.zeros((nChannels, nEvents))
        W_CM = np.zeros((nChannels, 2, nEvents))
        for i in range(nEvents):
            w_data = trainData[:, :, i, :]
            w1 = self.trca_matrix(w_data)
            w[:,i] = w1[:,0]
            W_CM[:, :, i] = w1

        # get mean temps
        mean_temp = np.zeros((nChannels, nTimes, nEvents))
        mean_temp = np.mean(trainData, -1)

        return w, W_CM, mean_temp


    def TRCA_test(self, testData, w, mean_temp, ensemble):
        """
        input:
        test_data of singe trial (Num of channels * num of sample points * number of events) 
        w （Num of channels * Num of events）
        mean_temp （Num of channels *Num of sample points * Num of events）
        ensemble : True of False 
        
        output:
        predict of singe trial
        """
        nEvents = mean_temp.shape[2]
        try:
            [nChannels, nTimes, nTrial] = testData.shape
        except:
            nTrial = 1
        if nTrial == 1:
            rr = np.zeros((nTrial, nEvents))
            for m in range(nTrial):     # the m-th test data
                r = np.zeros(nEvents)
                for n in range(nEvents):
                    test = testData
                    train = mean_temp[:, :, n]
                    if ensemble is True:
                        r[n] = corr2(train.T @ w, test.T @ w)
                    else:
                        r[n] = corr2(train.T @ w[:, n], test.T @ w[:,n])
                rr[m, :] = r
                
        else:
            rr = np.zeros((nTrial, nEvents))
            for m in range(nTrial):     # the m-th test data
                r = np.zeros(nEvents)
                for n in range(nEvents):
                    test = testData[:, :, m]
                    train = mean_temp[:, :, n]
                    if ensemble is True:
                        r[n] = corr2(train.T @ w, test.T @ w)
                    else:
                        r[n] = corr2(train.T @ w[:, n], test.T @ w[:,n])
                rr[m, :] = r
        return rr


class R_BL:
    def __init__(self):
        self.standardscaler = StandardScaler()
        pass

    def R_BL_matrix1(self, data, P):
        """
        Task-related component analysis (TRCA)
        input:
        X : eeg data (Num of channels * num of sample points * classes * number of trials)
        
        output:
        w: spatial filter
        """
        X = data
        nchans = X.shape[0]
        nTimes = X.shape[1]
        nClass = X.shape[2]
        nTrial = X.shape[3]
        # solves T
        T = np.mean(X, -1)
        # solves S and Q
        S = np.zeros((nchans, nchans))
        Q = np.zeros((nchans, nchans))
        for trigernum in range(nClass):
            for trialnum in range(nTrial):
                X1 = np.dot(X[:, :, trigernum, trialnum], P[trigernum, :, :])
                X2 = X[:, :, trigernum, trialnum]
                S = S + X1 @ (T[:, :, trigernum] @ P[trigernum, :, :]).T
                Q = Q + np.dot((X1 - X2), (X1 - X2).T)
        S = S/(nClass)
        Q = Q/(nTrial*nClass)
        b = np.dot(np.linalg.inv(Q), S)
        [eig_value, eig_w] = np.linalg.eig(b)
        # Descending order
        d = eig_value.argsort()[::-1]
        eig_w = eig_w[:, eig_value.argsort()[::-1]] # return indices in ascending order and reverse
        eig_value.sort()
        eig_value = eig_value[::-1] # sort in descending
        w = eig_w[:,0:2]
        return w.real
    

    def R_BL_matrix(self, data, P):
        """
        Task-related component analysis (TRCA)
        input:
        X : eeg data (Num of channels * num of sample points * classes * number of trials)
        
        output:
        w: spatial filter
        """
        X = data
        nchans = X.shape[0]
        nTimes = X.shape[1]
        nTrial = X.shape[2]
        # solves T
        T = np.mean(X, -1)
        # solves S and Q
        S = np.zeros((nchans, nchans))
        Q = np.zeros((nchans, nchans))

        for trialnum in range(nTrial):
            X1 = np.dot(X[:, :, trialnum], P)
            X2 = X[:, :, trialnum]
            S = S + X1 @ (T @ P).T
            # X22 = X2.reshape([nchans, nTimes * nTrial], order = 'F')
            # a = numpy.matlib.repmat(np.mean(X22,1), X22.shape[1], 1)
            # X22 = X22 - a.T
            # Q = Q + X22 @ X22.T
            Q = Q + np.dot((X1 - X2), (X1 - X2).T)

        b = np.dot(np.linalg.inv(Q), S)
        [eig_value, eig_w] = np.linalg.eig(b)
        # Descending order
        eig_w = eig_w[:, eig_value.argsort()[::-1]] # return indices in ascending order and reverse
        eig_value.sort()
        eig_value = eig_value[::-1] # sort in descending
        w = eig_w[:,0]
        return w.real
    

    def R_BL_train(self, trainData, P):
        """
        input:
        train_data  (Num of channels * num of sample points * num of events * number of train trials)
        P: 正交投影矩阵（num of events * num of sample points * num of sample points）
        output:
        w （Num of channels * 2）
        mean_temp （Num of channels *Num of sample points * Num of events）
        """
        [nChannels, nTimes, nEvents, nTrials] = trainData.shape
        # get w of event class
        # w = np.zeros((nChannels, nChannels))
        # for i in range(nEvents):
        #     w_data = trainData[:, :, i, :]
        #     w1 = self.R_BL_matrix(w_data,P[i,:,:])
        #     w[:,i] = w1
        w = np.zeros((nChannels, 2))
        w_data = trainData
        w = self.R_BL_matrix1(w_data, P)
        # get mean temps
        mean_temp = np.zeros((nChannels, nTimes, nEvents))
        mean_temp = np.mean(trainData, -1)
        return w, mean_temp
    

    def R_BL_test(self, testData, w, mean_temp, ensemble):
        """
        input:
        test_data of singe trial (Num of channels * num of sample points * number of events) 
        w （Num of channels * Num of events）
        mean_temp （Num of channels *Num of sample points * Num of events）
        ensemble : True of False 
        
        output:
        predict of singe trial
        """
        nEvents = mean_temp.shape[2]
        try:
            [nChannels, nTimes, nTrial] = testData.shape
        except:
            nTrial = 1
        if nTrial != 1:
            rr = np.zeros((nTrial, nEvents))
            for m in range(nTrial):     # the m-th test data
                r = np.zeros(nEvents)
                for n in range(nEvents):
                    test = testData[:, :, m]
                    train = mean_temp[:, :, n]
                    if ensemble is True:
                        r[n] = corr2(train.T @ w, test.T @ w)
                    else:
                        r[n] = corr2(train.T @ w[:, n], test.T @ w[:,n])
                rr[m, :] = r
        else:
            rr = np.zeros((nTrial, nEvents))
            for m in range(nTrial):     # the m-th test data
                r = np.zeros(nEvents)
                for n in range(nEvents):
                    test = testData
                    train = mean_temp[:, :, n]
                    if ensemble is True:
                        r[n] = corr2(train.T @ w, test.T @ w)
                    else:
                        r[n] = corr2(train.T @ w[:, n], test.T @ w[:,n])
                rr[m, :] = r

        return rr



class GetFeature:
    def __init__(self):
        pass

    def get_feature(self, data, T, w):
        '''
        input:
        data: get featurn of X, shape(nChans, nTimes, nTrail)
        T: template data of n class, (nChans, nTimes)
        w: filter of space: (nChans,2) or (nChans, nchans)
        return: feature , shape(nTrail, w.shape[1])
        '''
        try:
            nTrail = data.shape[2]
        except:
            nTrail = 1

        if nTrail == 1:
            feature = np.zeros((nTrail, w.shape[1]))
            for i in range(nTrail):
                X = data[:, :]
                T_1 = w[:, 0].T @ T
                T_2 = np.diff(T_1)
                X_1 = w[:, 0].T @ X
                X_2 = np.diff(X_1)
                feature[i, 0] = T_1 @ X_1.T
                feature[i, 1] = T_2 @ X_2.T
        else:
            feature = np.zeros((nTrail, w.shape[1]))
            for i in range(nTrail):
                X = data[:, :, i]
                T_1 = w[:, 0].T @ T
                T_2 = np.diff(T_1)
                X_1 = w[:, 0].T @ X
                X_2 = np.diff(X_1)
                feature[i, 0] = T_1 @ X_1.T
                feature[i, 1] = T_2 @ X_2.T
        return feature


    def get_feature_diff_coxx(self, data, T, w):
        '''
        input:
        data: get featurn of X, shape(nChans, nTimes, nTrail)
        T: template data of n class, (nChans, nTimes)
        w: filter of space: (nChans,2) or (nChans, nchans)
        return: feature , shape(nTrail, w.shape[1])
        '''
        try:
            nTrail = data.shape[2]
        except:
            nTrail = 1

        if nTrail == 1:
            # feature = np.zeros((nTrail, w.shape[1]*2))
            feature = np.zeros((nTrail, 2))
            for i in range(nTrail):
                for j in range(1):
                    X = data[:, :]
                    T_1 = w[:, j].T @ T
                    T_2 = np.diff(T_1)
                    X_1 = w[:, j].T @ X
                    X_2 = np.diff(X_1)
                    feature[i, j*2] = T_2 @ X_2.T
                    feature[i, j*2+1] = T_1 @ X_1.T
                    # feature[i, j] = T_1 @ X_1.T
        else:
            # feature = np.zeros((nTrail, w.shape[1]*2))
            feature = np.zeros((nTrail, 2))
            for i in range(nTrail):
                for j in range(1):
                    X = data[:, :, i]
                    T_1 = w[:, j].T @ T
                    T_2 = np.diff(T_1)
                    X_1 = w[:, j].T @ X
                    X_2 = np.diff(X_1)
                    feature[i, j*2] = T_2 @ X_2.T
                    feature[i, j*2+1] = T_1 @ X_1.T
                    # feature[i, j] = T_1 @ X_1.T
        return feature


class Classfication_Method(GetFeature):
    def __init__(self):
        GetFeature.__init__(self)
        self.RBL = R_BL()
        self.TRCA = eTRCA()


    def IC_NC_trainSVM(self, IC_traindata, NC_traindata, T, w, method = 'RBL'):
        '''
        input:
        IC_traindata: IC_traindata of x class, shape(nChans, nTimes, nTrial)
        NC_traindata: NC_traindata, shape(nChans, nTimes, nTrial)
        label: shape(2*nTrial, )
        method: 'RBL' or 'eTRCA'
        return: model of svm      
        '''
        nTrial = IC_traindata.shape[2]
        train_data = np.concatenate((IC_traindata, NC_traindata), axis=2)
        if method == 'RBL':
            train_feature = self.get_feature_diff_coxx(train_data, T, w)
        elif method == 'TRCA':
            train_feature = self.get_feature_diff_coxx(train_data, T, w)
        else:
            raise ValueError('Do not select right method ！')

        # self.standardscaler.fit(train_feature)
        # train_feature_standard = self.standardscaler.transform(train_feature)
        IC_label = np.ones((nTrial, 1))
        NC_lable = np.zeros((nTrial, 1))
        label = np.concatenate((IC_label, NC_lable), axis=0)


        model_SVM = svm.SVC(C=1, kernel='linear')
        model_SVM.fit(train_feature,label)
        print(model_SVM.score(train_feature, label))
        # draw = draw_fig_SVM()
        # draw.Visual_SVM(feature=train_feature, label= label, models = model_OneClassSVM, title='SVC(线性核)')
        return model_SVM


    def IC_NC_trainLDA(self, IC_traindata, NC_traindata, T, w, method = ''):
        '''
        input:
        IC_traindata: IC_traindata of x class, shape(nChans, nTimes, nTrial)
        NC_traindata: NC_traindata, shape(nChans, nTimes, nTrial)
        label: shape(2*nTrial, )
        method: 'RBL' or 'TRCA'
        return: model of svm
        '''
        nTrial = IC_traindata.shape[2]
        train_data = np.concatenate((IC_traindata, NC_traindata), axis=2)
        if method == 'RBL':
            train_feature = self.get_feature_diff_coxx(train_data, T, w)
        elif method == 'TRCA':
            train_feature = self.get_feature_diff_coxx(train_data, T, w)
        else:
            raise ValueError('Do not select right method ！')

        IC_label = np.ones((1,nTrial))
        NC_lable = np.zeros((1,nTrial))
        label = np.concatenate((IC_label, NC_lable), axis=1)[0]

        model_LDA = LDA()
        model_LDA.fit(train_feature, label)
        print(model_LDA.transform(train_feature))

        return  model_LDA




    def IC_NC_trainKNN(self, IC_traindata, NC_traindata, T, w):
        '''
        input:
        IC_traindata: IC_traindata of x class, shape(nChans, nTimes, nTrial)
        NC_traindata: NC_traindata, shape(nChans, nTimes, nTrial)
        label: shape(2*nTrial, )
        return: model of svm
        '''
        nTrial = IC_traindata.shape[2]

        train_data = np.concatenate((IC_traindata, NC_traindata), axis=2)

        train_feature = self.RBL_feature(train_data, T, w)

        IC_label = np.ones((nTrial,1))
        NC_lable = np.zeros((nTrial,1))

        label = np.concatenate((IC_label, NC_lable), axis=0)

        model_KNN = KNeighborsClassifier()
        model_KNN.fit(train_feature, label)
        print(model_KNN.score(train_feature, label))

        return model_KNN






        # IC_train_feature = self.RBL_feature(IC_traindata, T, w)
        # NC_train_feature = self.RBL_feature(NC_traindata, T, w)

        # model_OneClassSVM1 = svm.OneClassSVM(kernel='rbf', gamma='auto', nu =0.005)
        # model_OneClassSVM1.fit(IC_train_feature)
        # a = model_OneClassSVM1.predict(IC_train_feature)
        # IC_index = np.where(a==1)[0]
        # IC_feature = IC_train_feature[IC_index]

        # model_OneClassSVM2 = svm.OneClassSVM(kernel='rbf', gamma='auto', nu =0.005)
        # model_OneClassSVM2.fit(NC_train_feature)
        # a = model_OneClassSVM2.predict(NC_train_feature)
        # NC_index = np.where(a==1)[0]
        # NC_feature = NC_train_feature[NC_index]

        # train_feature = np.concatenate((IC_feature, NC_feature), axis=0)
        # IC_label = np.ones((IC_feature.shape[0],1))
        # NC_lable = np.zeros((NC_feature.shape[0],1))
        # label = np.concatenate((IC_label, NC_lable), axis=0)s

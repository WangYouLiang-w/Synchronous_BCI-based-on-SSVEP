import numpy as np
from PreProcessing_offline import PreProcessing
from algorithms import eTRCA, R_BL
from algorithms import Classfication_Method
import logging
import datetime
import os

#%%
subject_name = 'wyl2'
subject_filename = 'offline'
log_file_name = subject_name + '_R_BL'+'_offline'
if os.path.exists('./offline_log_tmp/'+log_file_name) == False:
    os.mkdir('./offline_log_tmp/'+log_file_name)
time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

logging.basicConfig(level=logging.DEBUG,
                    format = '%(asctime)s.%(msecs)03d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt = '%H:%M:%S',
                    filename = './offline_log_tmp/'+log_file_name+'/{}_{}.log'.format(subject_name, time_now),
                    filemode = 'w'
                    )
logger_obj = logging.getLogger('offline_log')

'''setting'''
Fres = [11.5, 12, 12.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 0]  # 对刺激块频率的对应关系，做了更改
# Fres = [ 8, 8.5, 9, 9.5, 10, 10.5, 11,11.5, 12, 12.5, 0]
Phas = [0.35, 0.70, 1.05, 1.40, 1.75, 2.10, 2.45, 2.80, 3.15, 3.5, 0]
CHANNELS = ['P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']

def alg_test(filepath, t_task, nclasses, sfreq=1000, num_filter=1, n_harmonics=5,addloop = 1):
    #setting
    chans = len(CHANNELS)
    delay = 0.14
    stimlength = int(t_task * sfreq)
    OVERLAP_TRIALS = addloop
    w_num = 2
    blocks = 2
    # load_data ={}
    # for block in range(blocks):
    preEEG = PreProcessing(filepath, t_begin = delay, t_end = delay + t_task, n_classes=nclasses,fs_down = sfreq, chans=CHANNELS,num_filter= num_filter)
    raw_data = preEEG.load_cnt_data()
    w_pass_2d = np.array([[8], [60]])
    w_stop_2d = np.array([[6], [62]])
    filtered_data = preEEG.filtered_data_iir(w_pass_2d, w_stop_2d, raw_data)

    # for idx_block in range(blocks):
    #     filtered_data = np.concatenate((load_data['bank' + str(idx_block)], load_data['bank' + str(idx_block)],load_data['bank' + str(idx_block)], load_data['bank' + str(idx_block)]), axis=3)

    #构造模板
    Yf, Pf= preEEG.generate_cca_references(Fres, sfreq, t_task, Phas, n_harmonics)
    #Cross-validation parameter setting
    nBlock = filtered_data.shape[4]
    n_splits = 4  # 4-fold
    trainslice, testslice = preEEG.ShuffleSplit_mean(nBlock, n_splits)

    control_mat = np.zeros((nclasses, nclasses))
    alg = R_BL()
    # train : get ensembleW of banks
    for split in range(n_splits):
        # train:get ensemble of banks
        train_w = np.zeros((num_filter, chans, w_num))
        train_meantemp = np.zeros((num_filter, chans, stimlength, nclasses))
        train_list = np.nonzero(trainslice[:,split])[0]
        train_data = filtered_data[:, :, :, :, train_list]

        for idx_filter in range(num_filter):
            TrainData = train_data[idx_filter, :, :, :, :] # n_channel * n_times * n_events * n_trials
            # train
            w, mean_temp = alg.R_BL_train(TrainData, Pf)
            train_w[idx_filter, :, :] = w
            train_meantemp[idx_filter, :, :, :] = mean_temp

        # test:
        test_list = np.nonzero(testslice[:,split])[0]
        test_data = filtered_data[:, :, :, :, test_list] # n_filters * n_channel * n_times * n_events * n_trials
        coef_mat = np.zeros((nclasses, OVERLAP_TRIALS))

        for triggernum in range(test_data.shape[3]):   # nEvents
            for trialnum in range(test_data.shape[4]): # nTrials
                trial_temp = np.mod(trialnum+1,OVERLAP_TRIALS)
                rrall = np.zeros((nclasses, 1))

                for idx_filter in range(num_filter):
                    test_trial = test_data[idx_filter, :, :, triggernum, trialnum]
                    rr = alg.R_BL_test(test_trial, train_w[idx_filter,:,:], train_meantemp[idx_filter, :, :, :], True)
                    if num_filter != 1:
                        rrall += (np.multiply(np.sign(rr), (rr**2)) * ((idx_filter+1) ** (-1.25) + 0.25)).T
                    else:
                        rrall += rr.T
                coef_mat[:, trial_temp] = rrall[:,0]

                if trialnum >= OVERLAP_TRIALS - 1 :
                    # 检测目标频率 f
                    coef = np.sum(coef_mat,1)
                    f = np.argmax(coef)
                    result = f
                    # 统计正确率
                    control_mat[triggernum, result] += 1

    acc = np.trace(control_mat) / ((nclasses)*(test_data.shape[4]-OVERLAP_TRIALS+1)*n_splits)

    print('acc = {}, OVERLAP_TRIALS = {}'.format( acc, OVERLAP_TRIALS))
    logger_obj.info('acc = {}, OVERLAP_TRIALS = {}'.format( acc, OVERLAP_TRIALS))
    return filtered_data, Pf


def alg_train(filtered_data, Pf):

    num_filter = filtered_data.shape[0]
    chans = filtered_data.shape[1]
    stimlength = filtered_data.shape[2]
    nclasses = filtered_data.shape[3]
    w_num = 2

    #训练
    alg = R_BL()
    CM = Classfication_Method()

    train_w = np.zeros((num_filter, chans, w_num))
    train_meantemp = np.zeros((num_filter, chans, stimlength, nclasses-1))

    for idx_filter in range(num_filter):
        TrainData = filtered_data[idx_filter, :, :, :, :] # n_channel * n_times * n_events * n_trials
        # train
        w, mean_temp = alg.R_BL_train(TrainData, Pf)
        train_w[idx_filter, :, :] = w
        train_meantemp[idx_filter, :, :, :] = mean_temp

    return train_w, train_meantemp

#%%
if __name__ == "__main__":
    t_task = 0.5
    # 离线测试
    filepath = os.path.join(r'subjects/'+ subject_name + '/', subject_filename + '.cnt')
    filter_data, Pf = alg_test(filepath, t_task=t_task, nclasses = 10, sfreq=1000, num_filter=1, n_harmonics = 5, addloop =2)
    # 离线训练
    train_w, train_meantemp, train_classifer = alg_train(filter_data, Pf)
    # 保存数据
    print('Template of shape:{0}'.format((train_meantemp.shape)))
    print('W of shape:{0}'.format((train_w.shape)))
    np.save('./data/Template.npy', train_meantemp)
    np.save('./data/W.npy', train_w)
    np.save('./data/P.npy', Pf)
    print('保存完成！')


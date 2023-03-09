import numpy as np
from algorithmInterface import algorithmthread
import os
from PreProcessing_offline import PreProcessing
import algorithms
import logging
import datetime

def ITR(n, p, t):
    if p == 1:
       itr = np.log2(n) * 60 / t
    else:
        itr = (np.log2(n) + p*np.log2(p) + (1-p)*np.log2((1-p)/(n-1))) * 60 / t

    return itr

subject_name = 'wyl2'
subject_filename = 'SimOnline'
log_file_name = subject_name + '_TRCA'+'_simonline'
if os.path.exists('./SimOnline_log_tmp/'+log_file_name) == False:
    os.mkdir('./SimOnline_log_tmp/'+log_file_name)
time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

logging.basicConfig(level=logging.DEBUG,
                    format = '%(asctime)s.%(msecs)03d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt = '%H:%M:%S',
                    filename = './SimOnline_log_tmp/'+log_file_name+'/{}_{}.log'.format(subject_name, time_now),
                    filemode = 'w'
                    )
logger_obj = logging.getLogger('simonline_log')

if __name__ == '__main__':
    #-------脑电处理的模板导入-----------#
    # load data
    filepath = './data/'
    # 获得空间滤波器
    w = np.load(filepath+'W.npy')
    # 获得模板信号
    template = np.load(filepath+'Template.npy')

    # 模拟在线测试
    filepath = os.path.join(r'subjects/'+ subject_name + '/', subject_filename + '.cnt')
    delay = 0.14
    t_task = 0.5
    nclasses = 10
    srate = 1000
    CHANNELS = ['P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']
    num_filter = 1
    OVERLAP_TRIALS = 2

    dataRunner = algorithmthread(fs=srate, w=w, template=template, addloop=OVERLAP_TRIALS, Nfb=1, mode='trca')

    preEEG = PreProcessing(filepath, t_begin = delay, t_end = delay + t_task, n_classes=nclasses,fs_down = srate, chans=CHANNELS,num_filter= num_filter)
    raw_data = preEEG.load_cnt_data()

    # test:
    coef_mat = np.zeros((nclasses, OVERLAP_TRIALS))
    rrall = np.zeros((nclasses, 1))
    control_mat = np.zeros((nclasses, nclasses))

    for triggernum in range(raw_data.shape[2]):   # nEvents
        for trialnum in range(raw_data.shape[3]): # nTrials
            test_data = raw_data[:, :, triggernum, trialnum]
            rrall,filtdata = dataRunner.extractor.test_algorithm(test_data)
            trial_temp = np.mod(trialnum+1,OVERLAP_TRIALS)
            coef_mat[:, trial_temp] = rrall[:,0]

            if trialnum >= OVERLAP_TRIALS - 1 :
                # 检测目标频率 f
                coef = np.sum(coef_mat,1)
                f = np.argmax(coef)
                result = f
                # 统计正确率
                control_mat[triggernum, result] += 1

    acc = np.trace(control_mat) / ((nclasses)*(raw_data.shape[3]-OVERLAP_TRIALS+1))
    itr = ITR(nclasses, acc, t_task)
    print('acc = {}, ITR = {}, OVERLAP_TRIALS = {}'.format( acc, itr,OVERLAP_TRIALS))
    logger_obj.info('acc = {}, ITR = {}, OVERLAP_TRIALS = {}'.format( acc, itr,OVERLAP_TRIALS))

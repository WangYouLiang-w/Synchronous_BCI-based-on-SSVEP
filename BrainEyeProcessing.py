import numpy as np
import mne
import scipy.io as scio
import json
import os
from algorithmInterface import algorithmthread

class BE_Processing:
    def __init__(self, stimlength, delay,filepath_online_brain_data=None, filepath_online_eye_data=None,Nchan = 21, fs_B = 1000, fs_E = 100):
        self.filepath_online_brain_data = filepath_online_brain_data
        self.filepath_online_eye_data = filepath_online_eye_data
        self.fs_B = fs_B
        self.fs_E = fs_E
        self.stimlength = stimlength
        self.delay = delay
        self.Nchan = Nchan


    def load_online_brain_data(self):
        '''
        获取在线脑电数据
        :return: raw_data shape(Nc,NTimes,triggernum)
        '''
        raw = mne.io.read_raw_cnt(self.filepath_online_brain_data, eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'], preload=True, verbose=False)
        num = 255
        mapping_ssvep = dict()
        for idx_command in range(127, num+1):
            mapping_ssvep[str(idx_command)] = idx_command
            events_ssvep, events_ids_ssvep = mne.events_from_annotations(raw, event_id=mapping_ssvep)
            pass

        data, times = raw[:]
        nTimes = int(self.stimlength * self.fs_B)
        triggernum = events_ssvep.shape[0]
        Nc = self.Nchan

        triggerpos = events_ssvep[:, 0]
        raw_data = np.zeros((Nc, nTimes, triggernum))

        for trigger_i in range(triggerpos.shape[0]-1):
            currenttriggerpos = triggerpos[trigger_i]
            raw_data[:, :, trigger_i] = data[:, int(currenttriggerpos+self.delay*self.fs_B+2):int(currenttriggerpos+(self.delay+self.stimlength)*self.fs_B+2)]*1000000

        return raw_data


    def load_online_eye_data(self):
        '''
        获取在线眼动数据
        :return: raw_data, shape(2,NTimes,triggernum)
        '''
        EYE = scio.loadmat(self.filepath_online_eye_data)    #数据
        data = EYE['eye_data'][:,:6,0]   # (X_T, Y_T, X, Y, Label, TimeStamp)
        event_line = EYE['event_data'][:,:,0]
        eye_data = data[:,:2].T # shape(2,Nsample)

        triggertype = data[:, 4].T
        triggerpos =  triggertype[np.where(triggertype==1)]
        triggernum = triggerpos.shape[0]

        Nc = 2
        stimlength = self.t_end - self.t_begin
        nTimes = int(stimlength * self.fs_E)
        raw_data = np.zeros((Nc, nTimes, triggernum))

        for trigger_i in range(triggernum):
            currenttriggerpos = triggerpos[trigger_i]
            raw_data[:, :, trigger_i] = eye_data[:, int(currenttriggerpos+1):int(currenttriggerpos+self.stimlength*self.fs_E+1)]

        return  raw_data


if __name__ == '__main__':
    #-------脑电处理的模板导入-----------#
    # load data
    filepath = './data/'
    # 获得空间滤波器
    w = np.load(filepath+'W.npy')
    # 获得模板信号
    template = np.load(filepath+'Template.npy')
    # 获取正交投影矩阵
    P = np.load(filepath+'P.npy')

    savedata = np.zeros((21, 500, 100))
    i = 0
    flagstop = False
    n_chan = 21                       # 采用的通道数
    srate = 1000                      # 采样频率
    brain_time_buffer = 10 # second   # 数据buffer
    epochlength = int(srate*0.64)     # 数据长度
    delay = int(srate*0.14)           # 延迟时间
    addloop = 2# 轮次
    trigger_flag = 0

    #通信设置#
    IP = '192.168.56.3'
    ControlAddr = (IP,7820)    # 控制单元的服务端地址
    FeedBackAddr = (IP,7830)   # 反馈单元的服务端地址
    dataRunner = algorithmthread(fs=srate, w=w, template=template, addloop=addloop, Nfb=1,
                             ControlAddr=ControlAddr, FeedBackAddr=FeedBackAddr, mode='R_LB')

    # 屏幕中刺激块的坐标导入
    presettingfile = open('StimPos.json')
    PreSetting= json.load(presettingfile)
    local_position = PreSetting['position']
    dataRunner.EyeTracker.recv_local_Json(local_position)

    #-------主程序------------------#
    # 读取在线脑电和眼动数据
    subject_name = 'wyl/run1'
    brain_filename = 'online'
    eye_filename = 'EYE_Online'
    brain_data = os.path.join(r'subjects/'+ subject_name + '/', brain_filename + '.cnt')
    # eye_data = os.path.join(r'subjects/'+ subject_name + '/', eye_filename + '.mat')
    BEDataRunner = BE_Processing(stimlength=0.5,delay=0.14,filepath_online_brain_data=brain_data)
    brain_raw_data = BEDataRunner.load_online_brain_data()  # shape(Nc,NTimes,triggernum)
    # eye_raw_data = BEDataRunner.load_online_eye_data()      # shape(2,NTimes,triggernum)
    # 标签数量
    Ntrigger = brain_raw_data.shape[2]
    # 处理主程序
    nFP = 0
    nTP = 0
    NIdle = 0
    NControl = 0
    Nright = 0
    for triggger_i in range(Ntrigger):
        # eye_test_data = eye_raw_data[:, :, triggger_i]
        brain_test_data = brain_raw_data[:, :, triggger_i]
        # dataRunner.RecvEyeData(eye_test_data)
        dataRunner.recvData(brain_test_data)

        r,__ = dataRunner.extractor.test_algorithm(dataRunner.testdata)
        dataRunner.appendResult(r)
        brain_result = dataRunner.BrainResultDecide()
        # eye_result = dataRunner.EyeResultDecide(dataRunner.eye_testdata)
        print("bain_result is {}".format(brain_result))
    #     if eye_result == 10:
    #         # 处于非控制态
    #         if brain_result != 10:
    #             nFP += 1
    #         else:
    #             Nright += 1
    #         NIdle += 1
    #
    #     else:
    #         if eye_result == brain_result:
    #             nTP += 1
    #             Nright += 1
    #
    #         NControl += 1
    #
    # FPR = nFP/NIdle
    # TPR = nTP / NControl
    # ACC = Nright / Ntrigger
    # min = (Ntrigger - 1) * 0.5
    # print('nTP:{}, nFP:{}, TPR:{}, FPR:{}, ACC:{}, Min(s):{}'.format(nTP, nFP, TPR, FPR, ACC, min))






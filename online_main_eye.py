import numpy as np
import json
from EyeDataServer import OnlineEyeDataRecver
from algorithmInterface import algorithmthread


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
    # 获取分类器
    Classifers = np.load(filepath+'classifer.npy', allow_pickle=True)

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
    dataRunner = algorithmthread(fs=srate, w=w, template=template, classifier= Classifers, addloop=addloop, Nfb=1,
                             ControlAddr=ControlAddr, FeedBackAddr=FeedBackAddr, mode='R_LB')

    #---------眼动设置-----------------#
    eye_fs = 100
    packet_length = 0.04
    eye_time_buffer = 1
    eye_datalength = int(eye_fs*0.5)

    # 屏幕中刺激块坐标的导入
    presettingfile = open('StimPos.json')
    PreSetting= json.load(presettingfile)
    local_position = PreSetting['position']
    dataRunner.EyeTracker.recv_local_Json(local_position)

    '''眼动线程'''
    thread_eyedata_sever = OnlineEyeDataRecver(eye_fs,packet_length, time_buffer=eye_time_buffer)
    thread_eyedata_sever.daemon = True
    thread_eyedata_sever.wait_connect()
    thread_eyedata_sever.start()

    label_flag = 0
    while not flagstop:
        data, eventline = thread_eyedata_sever.get_buffer_data()
        triggerPos = np.nonzero(eventline)[0]  # 找到非零元素的索引
        if triggerPos.shape[0] > 1:
            currentTriggerPos = triggerPos[-2]  # 取倒数第二个标签
            if data[:,currentTriggerPos+1:].shape[1]>=eye_datalength:
                if eventline[currentTriggerPos] != label_flag:   # 相邻的两个标签不同才决策
                    cutdata = data[:2, currentTriggerPos+1:]
                    epochdata = cutdata[:,0:eye_datalength]
                    print('E_Trigger name: {}, eye_shape as: {}'.format(eventline[currentTriggerPos], epochdata.shape))
                    dataRunner.RecvEyeData(epochdata)
                    dataRunner.run()
                    label_flag = eventline[currentTriggerPos]





from dataServer import dataserver_thread
import numpy as np
from EyeDataServer import OnlineEyeDataRecver
from algorithmInterface import algorithmthread
import json
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import atexit
import logging
import datetime


if __name__ == '__main__':
    choice = 0
    # subject_name = 'cwz'
    # subject_filename = 'online'
    # log_file_name = subject_name + '_R_BL'+'_online'
    # if os.path.exists('./Online_log_tmp/'+log_file_name) == False:
    #     os.mkdir('./Online_log_tmp/'+log_file_name)
    # time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    #
    # logging.basicConfig(level=logging.DEBUG,
    #                     format = '%(asctime)s.%(msecs)03d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    #                     datefmt = '%H:%M:%S',
    #                     filename = './Online_log_tmp/'+log_file_name+'/{}_{}.log'.format(subject_name, time_now),
    #                     filemode = 'w'
    #                     )
    # logger_obj = logging.getLogger('online_log')


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

    dataRunner = algorithmthread(fs=srate, w=w, template=template,addloop=addloop, Nfb=1,
                                 ControlAddr=ControlAddr, FeedBackAddr=FeedBackAddr, mode='R_LB')

     #--------------子线程--------------#
    '''脑电线程'''
    thread_data_server = dataserver_thread(time_buffer=brain_time_buffer)
    thread_data_server.Daemon = True
    notconnect = thread_data_server.connect_tcp()
    if notconnect:
        raise TypeError("Can't connect recoder, Please open the hostport")
    else:
        thread_data_server.start_acq()
        thread_data_server.start()
        print('Data server connected')

    #--------------主线程--------------#
    if choice == 0:
        B_label_flag = 0
        while not flagstop:
            if thread_data_server.stop_flag == 1:
                break
            data, eventline = thread_data_server.get_buffer_data()
            triggerPos = np.nonzero(eventline)[0]  # 找到非零元素的索引
            if triggerPos.shape[0] > 1:
                currentTriggerPos = triggerPos[-2]  # 取倒数第二个标签
                if data[:,currentTriggerPos+1:].shape[1]>=epochlength:
                    if eventline[currentTriggerPos] != B_label_flag:   # 相邻的两个标签不同才决策
                        cutdata = data[:, currentTriggerPos+1:]
                        epochdata = cutdata[:,delay:epochlength]
                        dataRunner.recvData(epochdata,eventline[currentTriggerPos])
                        print('B_Trigger name: {}, brain_shape as: {}'.format(eventline[currentTriggerPos], epochdata.shape))
                        dataRunner.run()
                        B_label_flag = eventline[currentTriggerPos]
    # print(dataRunner.results)






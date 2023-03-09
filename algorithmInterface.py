import numpy as np
from numpy import ndarray
import scipy.signal as signal
import scipy.io as scio
import socket
import time
from scipy.stats import pearsonr
import math
import algorithms
from PreProcessing import PreProcessing


class Kalman_filter:
    def __init__(self, length, fs):
        self.data_length = length
        self.Q = np.load('Q.npy')#np.zeros((4,4))
        self.A = np.array([[1,0,1/fs,0],[0,1,0,1/fs],[0,0,1,0],[0,0,0,1]])
        self.P = np.eye(4,dtype=int)
        self.X = np.array([[0.0],[0.0],[0],[0]])

        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.R = np.load('R.npy')#np.zeros((2,2))
        self.err = np.load('err.npy')#np.zeros((1,2))

        self.K = np.zeros((4,2))
        self.t = 1/fs
        #----
        self.W_screen = 1/2560         #注视点的归一化范围
        self.H_screen = 1/1440
        self.local_Setting = {}
        self.UScreen_position = []
        self.MScreen_position = []

        self.stim_position_U = []
        self.stim_position_M = []

        self.UI_Setting = {}
        self.UScreen_position1 = []
        self.MScreen_position1 = []

        self.stim_position_U1 = []
        self.stim_position_M1 = []

        self.max_gap_length=75
        self.fs = fs
        self.distance = []

        self.local_position = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}

        self.gaze_points = np.zeros((1,6))


    def recv_local_Json(self,stim_position):
        '''
        界面配置文件,将上下屏上的刺激块分到每个区域
        :param stim_position:{}  刺激块在屏幕中的像素坐标位置
        :return:self.stim_position_U:[list] 上屏划分区域后归一化后的坐标位置
                self.stim_position_M:[list] 下屏划分区域后归一化后的坐标位置
        '''
        self.local_position = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}
        # 接收界面配置Json文件
        self.local_Setting = stim_position
        # 获取当前配置文件中包含每个屏的信息
        for key in self.local_Setting :

            if key[0] == 'U':
                self.UScreen_position.append(key)
            elif key[0] == 'M':
                self.MScreen_position.append(key)

        # 获取每个屏幕上刺激块的位置
        for num_U in range(len(self.UScreen_position)):  # 1,2  ["",X,Y,Size]
            U_GazePoint = [0.5+self.local_Setting[self.UScreen_position[num_U]][2]*self.W_screen,
                                            1-self.local_Setting[self.UScreen_position[num_U]][3]*self.H_screen]

            self.stim_position_U.append(U_GazePoint)
            # 区域1
            # print(self.UScreen_position[num_U])
            if self.local_Setting[self.UScreen_position[num_U]][0] == '1':
                self.local_position['1'].append(U_GazePoint)
            # 区域2
            if self.local_Setting[self.UScreen_position[num_U]][0] == '2':
                self.local_position['2'].append(U_GazePoint)
            # 区域3
            if self.local_Setting[self.UScreen_position[num_U]][0] == '3':
                self.local_position['3'].append(U_GazePoint)

    
        for num_M in range(len(self.MScreen_position)):  # 1,2  ["",X,Y,Size]
            M_GazePoint = [1.5+self.local_Setting[self.MScreen_position[num_M]][2]*self.W_screen,
                                            1-self.local_Setting[self.MScreen_position[num_M]][3]*self.H_screen]
            self.stim_position_M.append(M_GazePoint)

            # 区域4
            if self.local_Setting[self.MScreen_position[num_M]][0] == '4':
                self.local_position['4'].append(M_GazePoint)

            # 区域5
            if self.local_Setting[self.MScreen_position[num_M]][0] == '5':
                self.local_position['5'].append(M_GazePoint)

            # 区域6
            if self.local_Setting[self.MScreen_position[num_M]][0] == '6':
                self.local_position['6'].append(M_GazePoint)

    
    def liner_insert(self,fixations):
        '''
        线性插值
        :param fixations:list
        :return:fixations:list
        '''
        # 小于interval_num 需要内插
        interval_num = (self.fs*self.max_gap_length)/1000
        fix_len = len(fixations)
        # FT定义为开始 TF定义为结束
        start_idx, end_idx = [], []
        for ii in range(fix_len-2):
            if (fixations[ii] != 0) & \
                (fixations[ii+1] == 0) :
                start_idx.append(ii)
            if (fixations[ii] == 0) & \
                (fixations[ii+1] != 0):
                end_idx.append(ii)
        for start, end in zip(start_idx, end_idx):
            nan_len = end-start
            if nan_len>interval_num:
                px = [fixations[start], fixations[end+1]]
                interx = ((px[1]-px[0])*np.arange(nan_len+1)/float(nan_len+1)+px[0]).tolist()
                for ii in range(1, len(interx)):
                    fixations[start+ii] = interx[ii]
        return fixations


    def get_eyedata(self,data):
        '''
        @ 获取眼动数据
        :param data: data (nparray).shape(4,n)
                     n:样本点数
        :return: eye_data (nparray).shape(n,3)
                 (x,y)
                 screen_flag:0 or 1 (0:上屏,1:下屏)
        '''
        eye_data = np.zeros((data.shape[1],3))
        for num in range(data.shape[1]):
            # print(data)
            eye_data[num,2] = data[3,num]/10000
            # print(eye_data[num,2])
            if eye_data[num,2] == 1.0:        # 上屏点
                gaze_right_eye = data[0, num]/10000
                gaze_left_eye = data[1, num]/10000
            else:
                gaze_right_eye = data[0, num]/10000 + 1
                gaze_left_eye = data[1, num]/10000 + 1

            eye_data[num,0:2] = [gaze_right_eye,gaze_left_eye]
            
        # 线性插值
        # eye_data[:,0] = self.liner_insert(eye_data[:,0])
        # eye_data[:,1] = self.liner_insert(eye_data[:,1])
        return eye_data

    
    def filter(self,Eyedata):
        '''
        @ 对数据进行卡尔曼滤波
        :param data: Eyedata [nparray].shape(4,n)
        :return:filter_data  [nparray].shape(n,3):[:,(x,y,flag)]
        '''
        data = self.get_eyedata(Eyedata)
        # data = Eyedata
        xy_data = data[:,0:2]
        pre_data = self.X
        P = self.P
        K = self.K
        # for i in range(xy_data.shape[0]):
        #     pre_datas = xy_data[i,:] - self.err     # 弥补上偏差
        #     measure_data = np.mat(pre_datas).T # 测量值
        #     P = np.dot(np.dot(self.A,P),self.A.T)               # + self.Q
        #     a = np.dot(np.dot(self.H,P),self.H.T) + self.R
        #     b = np.dot(P,self.H.T)
        #     K = np.dot(b,np.linalg.inv(a))
        #     filter_data = pre_data + np.dot(K,measure_data-np.dot(self.H,pre_data))
        #     pre_data = filter_data
        #     data[i,0:2] = np.array(filter_data[0:2,0].T)
        #     P = np.dot(np.eye(4)-np.dot(K,self.H),P)
        return data


    def normal_model(self,stim_pos,Eyedata):
        '''
        @构建卡尔曼滤波后数据距离分布模型,计算目标位置的概率
        :param data: stim_pos:[list]:[[x1,y1],[x2,y2],....,[xn,yn]]
        :param data: Eyedata (nparray).shape(4,n)
        :return: [P1(x),P2(x),...,Pn(x)]
                 n: 当前决策区域内的刺激块个数
        '''
        P = []
        filter_data  = self.filter(Eyedata)
        # a = np.nonzero(filter_data[:,0])
        x = 0
        x_count = 0
        y = 0
        y_count = 0
        print(len(filter_data[:,0]))
        # 去除零元素对注视中心的计算
        for i in range(len(filter_data[:,0])):
            if filter_data[i,0] != 0:
                x = x + filter_data[i,0]/self.W_screen
                x_count += 1

        for j in range(len(filter_data[:,1])):
            if filter_data[j,1] != 0:
                y = y + filter_data[j,1]/self.H_screen
                y_count += 1
        x_mean = x/x_count
        y_mean = y/y_count

        if len(stim_pos) != 0:
            for pos in range(len(stim_pos)):
                x = stim_pos[pos][0]/self.W_screen
                y = stim_pos[pos][1]/self.H_screen
                dis_xy = math.sqrt((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean))
                P.append(dis_xy)
            print('P:{}'.format(min(P)))
        else:
            P = None
        return P


class EyeProcess:

    def __init__(self):
        self.SuitPoint = 0
        # self.eyetrack_threshold = int(datalength*0.35)   # 所占比例0.4
        self.W_screen = 1/2560         #注视点的归一化范围
        self.H_screen = 1/1440
        self.local_Setting = {}
        self.stim_position = []
        self.gaze_points = np.zeros((1,2))


    def recv_local_Json(self,stim_position):
        '''
        界面配置文件,将上下屏上的刺激块分到每个区域
        :param stim_position:{}  刺激块在屏幕中的像素坐标位置
        :return:self.stim_position:[list] 刺激块在屏幕中归一化后的坐标位置
        '''
        # 接收界面配置Json文件
        self.local_Setting = stim_position
        # 获取当前配置文件中包含每个屏的信息
        for key in self.local_Setting :
        # 获取屏幕上刺激块的位置
            GazePoint = [self.local_Setting[key][0]*self.W_screen, self.local_Setting[key][0]*self.H_screen]
            self.stim_position.append(GazePoint)


    def EyeTrace_Position(self,eyedatas,LocalNum):
        '''
        将眼动数据分到相应的控制区域
        :param eyedatas:
        :param LocalNum: 0：非控制区域， 1：控制区域
        :return: self.gaze_points[0][LocalNum] 在各个区域上的有效注视点数目
        '''
        gaze_x_y = eyedatas
        # print("眼动追踪区域:{},模板数量:{}".format(LocalNum,self.template_num))
        if ((gaze_x_y[0] > 0.05 and gaze_x_y[0] < 0.95) and (gaze_x_y[1] > 0.05 and gaze_x_y[1] < 0.95)):
            self.gaze_points[0][LocalNum] = self.gaze_points[0][LocalNum] + 1


    def normal_model(self,stim_pos,Eyedata):
        '''
        @构建眼动数据距离分布模型,计算目标位置的概率
        :param data: stim_pos:[list]:[[x1,y1],[x2,y2],....,[xn,yn]]
        :param data: Eyedata (nparray).shape(2,n)
        :return: [P1(x),P2(x),...,Pn(x)]
                 n: 当前决策区域内的刺激块个数
        '''
        P = []
        x = 0
        x_count = 0
        y = 0
        y_count = 0
        # 去除零元素对注视中心的计算
        for i in range(len(Eyedata[0,:])):
            if Eyedata[0,i] != 0:
                x = x + Eyedata[0,i]/self.W_screen   #转到像素
                x_count += 1

        for j in range(len(Eyedata[1,:])):
            if Eyedata[1,j] != 0:
                y = y + Eyedata[1,j]/self.H_screen
                y_count += 1

        x_mean = x/x_count
        y_mean = y/y_count

        for pos in range(len(stim_pos)):
            x = stim_pos[pos][0]/self.W_screen
            y = stim_pos[pos][1]/self.H_screen
            dis_xy = math.sqrt((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean))
            P.append(dis_xy)
        print('P:{}'.format(min(P)))

        return P


    def EyeDecideApply(self,data):
        '''
        对刺激区域的眼动数据进行决策
        :param data: eyedata，shape(2,n)
        :return: eye_decide_result(0-9)
        '''
        P = self.normal_model(self.stim_position,data)
        # if P != None:
        #     if min(P) > 200:
        #         eye_decide_result = 10
        #     else:
        #         eye_decide_result = P.index(min(P))
        # else:
        #     eye_decide_result = 10
        eye_decide_result = P.index(min(P))

        return  eye_decide_result



    def OfflineResultEyeDecide(self,data):
        '''
        获取眼动数据的结果
        :param data: eyedata, shape(2,n)
        :return: eye_decide_result (0-9,10)
        '''
        self.gaze_points = np.zeros((1,2))
        for num in range(data.shape[1]):
            gaze_x= data[0, num]
            gaze_y = data[1, num]
            gaze_x_y = [gaze_x,gaze_y]

            # 判断眼动数据属于那个位置:0:非刺激区域， 1：刺激区域
            if ((gaze_x_y[0] > 0.5 and gaze_x_y[0] < 1) and (gaze_x_y[1] > 0 and gaze_x_y[1] < 0.5)):
                self.EyeTrace_Position(gaze_x_y,0)
            else:
                self.EyeTrace_Position(gaze_x_y,1)


        gaze_points_max = np.max(self.gaze_points)
        if gaze_points_max != 0:
            index = np.argmax(self.gaze_points)
            if index == 0:                          # 注视点主要在非刺激区域
                eye_decide_result = 10
            else:
                eye_decide_result = self.EyeDecideApply(data)
        else:
            eye_decide_result = 10

        return eye_decide_result


    def OnlineResultEyeDecide(self,data):
        '''
        获取眼动数据的结果
        :param data: eyedata, shape(2,n)
        :return: eye_decide_result (0-9,10)
        '''
        self.gaze_points = np.zeros((1,2))
        for num in range(data.shape[1]):
            gaze_x= data[0, num]/10000
            gaze_y = data[1, num]/10000
            gaze_x_y = [gaze_x,gaze_y]

            # 判断眼动数据属于那个位置:0:非刺激区域， 1：刺激区域
            if ((gaze_x_y[0] > 0.5 and gaze_x_y[0] < 1) and (gaze_x_y[1] > 0 and gaze_x_y[1] < 0.5)):
                self.EyeTrace_Position(gaze_x_y,0)
            else:
                self.EyeTrace_Position(gaze_x_y,1)


        gaze_points_max = np.max(self.gaze_points)
        if gaze_points_max != 0:
            index = np.argmax(self.gaze_points)
            if index == 0:                          # 注视点主要在非刺激区域
                eye_decide_result = 10
            else:
                eye_decide_result = self.EyeDecideApply(data)
        else:
            eye_decide_result = 10

        return eye_decide_result


# offer a common test interface to any algorithm.
class TRCA(PreProcessing):
    def __init__(self, fs, TRCA_spatialFilter, template, Nsub=1):
        PreProcessing.__init__(self, fs, Nsub)
        self.W = TRCA_spatialFilter     # W为TRCA空间滤波器，ndarray (Nsub, Nchannels, Ntarget)
        self.template = template        # template为模板，ndarray (Nsub,  Nchannels, Ntimes, Ntarget)
        self.Nsub = Nsub
        self.Nc = template.shape[1]
        self.Ntimes = template.shape[2]
        self.Ntarget = template.shape[3]
        self.TRCA = algorithms.eTRCA()


    def test_algorithm(self, rawdata):
        w_pass_2d = np.array([[8], [60]])
        w_stop_2d = np.array([[6], [62]])
        filtdata = np.zeros((self.Nsub, self.Nc, self.Ntimes))
        filtdata = self.filtered_data_iir(w_pass_2d=w_pass_2d, w_stop_2d=w_stop_2d, data=rawdata)
        rrall = np.zeros((self.Ntarget, 1))
        for sub_i in range(self.Nsub):
            rr = self.TRCA.TRCA_test(filtdata[sub_i, :, :], self.W[sub_i, :, :], self.template[sub_i, :, :, :],True)
            if self.Nsub != 1:
                rrall += (np.multiply(np.sign(rr), (rr**2)) * ((sub_i+1) ** (-1.25) + 0.25)).T
            else:
                rrall += rr.T
        return rrall, filtdata


class R_BL(PreProcessing):
    def __init__(self, fs, RBL_spatialFilter, template, Nsub=1):
        PreProcessing.__init__(self, fs, Nsub)
        self.W = RBL_spatialFilter     # W为R_BL空间滤波器，ndarray (Nsub, Nchannels, 2)
        self.template = template        # template为模板，ndarray (Nsub,  Nchannels, Ntimes, Ntarget)
        self.Nsub = Nsub
        self.Nc = template.shape[1]
        self.Ntimes = template.shape[2]
        self.Ntarget = template.shape[3]
        self.RBL = algorithms.R_BL()


    def test_algorithm(self, rawdata):
        w_pass_2d = np.array([[8], [60]])
        w_stop_2d = np.array([[6], [62]])
        filtdata = np.zeros((self.Nsub, self.Nc, self.Ntimes))
        filtdata = self.filtered_data_iir(w_pass_2d=w_pass_2d, w_stop_2d=w_stop_2d, data=rawdata)
        rrall = np.zeros((self.Ntarget, 1))
        for sub_i in range(self.Nsub):
            rr = self.RBL.R_BL_test(filtdata[sub_i, :, :], self.W[sub_i, :, :], self.template[sub_i, :, :, :],True)
            if self.Nsub != 1:
                rrall += (np.multiply(np.sign(rr), (rr**2)) * ((sub_i+1) ** (-1.25) + 0.25)).T
            else:
                rrall += rr.T
        return rrall, filtdata


class algorithmthread():
    def __init__(self, fs, w, template, addloop, Nfb, ControlAddr=None, FeedBackAddr=None, mode='trca'):
        self.sendresult = 0
        self.Nsub = Nfb
        self.template = template
        self.mode = mode
        if mode == 'trca':
            self.extractor = TRCA(fs,w, template, Nsub=Nfb)
        else:
            self.extractor = R_BL(fs, w, template, Nfb)
            self.W = w  # W_CM为状态检测时用的空间滤波器，ndarray (Nsub, Nchannels, 2)

        #-----------脑电部分---------#
        self.testdata = np.array([0])
        self.resultCount = np.zeros((template.shape[3],addloop))
        self.currentPtr = 0
        self.resultPtr = 0
        self.addloop = addloop
        #------------眼动部分-----------#
        self.EyeTracker = EyeProcess()
        # self.eye_testdata = np.array([0])
        # self.labels = 0
        # self.results = np.zeros((3,36))
        # self.result_count = 0
        # self.Send_command = 'IDLE'
        #-------------通信部分----------#
        self.sock_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.controlCenterAddr = ControlAddr
        self.sock_Feedback = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.FeedbackAddr = FeedBackAddr
        self.result = 0
        self.all_count = 0
        self.right_count = 0
        self.target = 0

    def recvData(self,rawdata,label = 0):
        '''接收待处理的脑电数据'''
        self.testdata = np.copy(rawdata)
        self.target = np.mod(label-1, 6)    #标签对应的脑电结果
        self.labels = label-1              # 标签值 0-10


    def RecvEyeData(self,rawdata):
        '''获取待处理的眼动数据'''
        self.eye_testdata = rawdata


    def run(self):
        r,__ = self.extractor.test_algorithm(self.testdata)
        self.appendResult(r)
        brain_result = self.BrainResultDecide()
        # eye_result = self.EyeResultDecide(self.eye_testdata)
        self.SenderRsult(brain_result)                         # 发送决策结果
        self.clearTestdata()                       # 清除数据缓存


    def appendResult(self,data):
        '''
        loop of iter add
        :param data: rawdata
        :return:
        '''
        self.resultCount[:,np.mod(self.currentPtr, self.addloop)] = data.reshape(10,)    # 存放的是相关系数 叠加轮次是5轮 0-4
        self.currentPtr += 1


    def BrainResultDecide(self):
        '''
        get result of BrainDecide
        :return:
        '''
        # 得到目标频率
        decide = np.sum(self.resultCount,axis=1,keepdims=False)
        brain_decide_result = np.argmax(decide)

        return brain_decide_result


    def EyeResultDecide(self, eye_testdata):
        '''
        get result of EyeDecide
        :return:
        '''
        eye_result = self.EyeTracker.OnlineResultEyeDecide(eye_testdata)
        return eye_result


    def SenderRsult(self,result):
        '''
        send command to UAV or UI
        :return:
        '''
        #self.SendCommand(result)
        self.SendFeedBack(result+1)


    def SendCommand(self, command):
        '''
        :param command: control of command，0-10
        :return:
        '''
        msg = bytes(str(command), "utf8")
        print('the result is :{}'.format(msg))
        self.sock_client.sendto(msg, self.controlCenterAddr)


    def SendFeedBack(self, command):
        '''
        send feedback to interface 0f stimulation
        :param command: feedback
        :return:
        '''
        msg = bytes(str(command), "utf8")
        print('the Feedback is :{}'.format(msg))
        self.sock_Feedback.sendto(msg, self.FeedbackAddr)


    def clearTestdata(self):
        self.testdata = np.array([0])
        self.eye_testdata = np.array([0])




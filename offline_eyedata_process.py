import numpy as np
from threading import Thread, currentThread
import scipy.io as scio

#%%
class EyeTracking_offline_process(Thread):
    def __init__(self,eye_data,event_data):
        Thread.__init__(self)
        self.sample_fre = 120     # 眼动仪的采样频率
        self.time_interval = 0.5  # 400ms
        self.datalength = int(self.sample_fre*self.time_interval)

        self.eye_data = eye_data
        self.event_data = event_data
        self.databuffer = np.zeros((6,self.datalength))
        self.chanel = 6
        self.trail_num = 15
        self.trigger_num = 12

        self.W_screen = 1/1920         #注视点的归一化范围        
        self.H_screen = 1/1080
        self.gaze_points = np.zeros((1,12))    # 注视块的数目

        self.position_stim = [(0.5+(0*self.W_screen),0.5-(325*self.H_screen)),(0.5+(-100*self.W_screen),0.5-(-325*self.H_screen)),
                (0.5+(325*self.W_screen),0.5-(325*self.H_screen)),(0.5+(-300*self.W_screen),0.5-(-325*self.H_screen)),
                (0.5+(-525*self.W_screen),0.5-(100*self.H_screen)),(0.5+(525*self.W_screen),0.5-(100*self.H_screen)),
                (0.5+(525*self.W_screen),0.5-(-100*self.H_screen)),(0.5+(-525*self.W_screen),0.5-(-100*self.H_screen)),
                (0.5+(-325*self.W_screen),0.5-(325*self.H_screen)),(0.5+(0*self.W_screen),0.5-(-125*self.H_screen)),
                (0.5+(300*self.W_screen),0.5-(-325*self.H_screen)),(0.5+(100*self.W_screen),0.5-(-325*self.H_screen))]            # 注视刺激块的范围
        
        self.SuitPoint = 0
        self.block_size = 150   # 块的大小
        self.eyetrack_threshold = int(self.datalength* 0.6)   # 所占比例0.6
        self.epoch = np.zeros((self.datalength,self.chanel,self.trail_num, self.trigger_num))
    
    
    def find_labelpos_eye_data(self,pos):
        '''寻找距离标签位置最近的眼动数据'''
        datanum = self.eye_data.shape[0]
        min_off = 10000000000000
        # 寻找距离标签位置最近的数据开始截取数据
        for i in range(datanum):
            if np.abs(self.eye_data[i,5]- pos) < min_off:
                min_off = np.abs(self.eye_data[i,5]- pos)
                pos_num = i
        #----按照采样率截取数据----#
        data = eye_data[pos_num:pos_num+60,:]
        self.databuffer = eye_data[pos_num:pos_num+60,:]

        #----按照系统时间戳截取数据------#
        return self.databuffer
        

    def cut_eye_data(self):
        '''截取眼动数据'''
        triggernum = self.event_data.shape[0]
        triggertype = np.zeros((triggernum,1))
        triggerpos = np.zeros((triggernum,1))
        
        for eventnum in range(triggernum):
            triggertype[eventnum] = self.event_data[eventnum,0]  # 标签值
            triggerpos[eventnum] = self.event_data[eventnum,2]   # 标签的系统时间戳位置
        
        uniquetrigger = np.unique(triggertype)
        uniquetriggernum = uniquetrigger.shape[0]

        for triggernum in range(uniquetriggernum):
            currenttriger = uniquetrigger[triggernum]
            currenttrigerpos = triggerpos[triggertype==currenttriger]
            for j in range(len(currenttrigerpos)):
                self.epoch[:,:,j,triggernum] = self.find_labelpos_eye_data(currenttrigerpos[j])
        
        return self.epoch


    def run(self):
        self.cut_eye_data()
        count = 0
        right_count = 0
        for triggernum in range(self.epoch.shape[3]):
          for trials in range(self.epoch.shape[2]):
              rl_eye_data = self.epoch[:, :, trials, triggernum]
              gaze_right_eye = [0, 0]
              gaze_left_eye = [0, 0]
              count = count + 1

              for num in range(self.epoch.shape[0]):
                  gaze_right_eye[0] = rl_eye_data[num, 0]
                  gaze_right_eye[1] = rl_eye_data[num, 1]
                  gaze_left_eye[0] = rl_eye_data[num, 2]
                  gaze_left_eye[1] = rl_eye_data[num, 3]

                # 双眼数据有效
                  if ((gaze_right_eye[0] > 0.05 and gaze_right_eye[0] < 0.95) and (gaze_right_eye[1] > 0.05 and gaze_right_eye[1] < 0.95)) and ((gaze_left_eye[0] > 0.05 and gaze_left_eye[0] < 0.95) and (gaze_left_eye[1] > 0.05 and gaze_left_eye[1] < 0.95)):
                      gaze_right_left_eyes = ((gaze_right_eye[0]+gaze_left_eye[0])/2, (gaze_right_eye[1]+gaze_left_eye[1])/2)
                      self.SuitPoint = self.SuitPoint + 1
                      k = 0
                      for position in self.position_stim:
                          if (gaze_right_left_eyes[0] > position[0]- self.block_size/1920 and gaze_right_left_eyes[0] < position[0]+ self.block_size/1920) and (gaze_right_left_eyes[1] > position[1]- self.block_size/1080 and gaze_right_left_eyes[1] < position[1]+ self.block_size/1080):
                              self.gaze_points[0][k] = self.gaze_points[0][k] + 1
                          k = k + 1

              self.SuitPoint = 0
              if max(self.gaze_points[0]) > self.eyetrack_threshold:
                  eye_decide_result = np.argmax(self.gaze_points[0])
                  self.gaze_points[0] = 0
                  if eye_decide_result == triggernum:
                      right_count = right_count + 1
              else:
                  eye_decide_result = 0


        acc = right_count/count
        print('ACC:{}'.format(acc))
        print(count)
        print(right_count)


#%%
if __name__ == '__main__':
    #---------读取眼动数据------------#
    subject_name = 'wyl'
    FilePath = './eye_data/'+subject_name+'/'
    EYE = scio.loadmat(FilePath+'EYE_Online.mat')    #数据
    eye_data = EYE['eye_data'][:,:6,0]
    event_data = EYE['event_data'][:,:,0]
    dataRunner = EyeTracking_offline_process(eye_data,event_data)  
    epoch = dataRunner.cut_eye_data()

    #%% 数据发送的线程
    dataRunner = EyeTracking_offline_process(eye_data,event_data)  
    dataRunner.daemon = True
    dataRunner.start()
    dataRunner.join()

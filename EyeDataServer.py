from concurrent.futures import thread
from pkgutil import get_data
import socket
from threading import Thread
import threading
import time
import numpy as np


class OnlineEyeDataRecver(Thread):
    """
    规定：
        数据包头部占4字节
        整型占8字节
        字符串长度位占2字节
        字符串不定长
    """
    CHANNELS = ['right_gaze_data[0]', 'left_gaze_data[1]', 'systime_stamp', 'label']

    def __init__(self, fs_orig = 100,packet_length=0.04,time_buffer = 0.5,ip_address = '127.0.0.1'):
        Thread.__init__(self)

        # 眼动数据分析属性
        self.fs_orig = fs_orig
        self.channels = len(self.CHANNELS)
        self.time_buffer = time_buffer
        self.n_points_buffer = int(np.round(fs_orig*time_buffer))
        self.data_buffer = np.zeros((self.channels,self.n_points_buffer))
        self.current_ptr = 0
        self.nUpdata = 0
        self.client_socket = None
        self.ip_address = ip_address
        self.port = 8848

        self.read_data_flag = False

        # 眼动数据解包分析属性
        self.dur_one_packet = packet_length
        self.n_points = int(np.round(fs_orig*self.dur_one_packet))
        self.packet_data_bytes = (self.channels*self.n_points)*8

        


    def wait_connect(self):
        '''
        Initialize TCP and Connect with EEG device.
        :return:
            self.s_client: object of socket.
        '''
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        SEND_BUF_SIZE = self.packet_data_bytes  # unit: bytes
        RECV_BUF_SIZE = self.packet_data_bytes * 9  # unit: bytes
        for i in range(5):

            try:
                time.sleep(1.5)
                self.client_socket.connect((self.ip_address, self.port))
                print('Eye Connect Successfully.')

                self.client_socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
                self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUF_SIZE)
                self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUF_SIZE)

                
                buff_size_send = self.client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                buff_size_recv = self.client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                print('Current eye recv buffer size is {} bytes, send buff size is {} bytes.'.format(buff_size_recv, buff_size_send))
                break
            except:
                print('没有有效连接')
                self.client_socket.close()
                

    def close_connection(self):
        self.client_socket.close()


    def unpack_header(self,header_bytes):
        '''解包头'''
        packet_size = int.from_bytes(header_bytes, byteorder='little')
        return packet_size
          

    def get_eye_data(self):
        #前四个字节是包的字节数，进行解包头
        header_bytes = self._recv_fixed_len(4)
        packet_size = self.unpack_header(header_bytes)
        
        if packet_size != self.packet_data_bytes:
            print('databytes have problem!!.')
            
        else:
            # 分解数据
            recv_data = self._recv_fixed_len(packet_size)
            re_data = []
            length = int(self.channels * self.n_points)
            for i in range(length):
                try:
                    ret = recv_data[:8]
                    recv_data = recv_data[8:]
                    val = int.from_bytes(ret, byteorder='little')
                    re_data.append(val)
                except:
                    raise Exception("数据异常！")
            
            try:
                new_data_trans = np.asarray(re_data).reshape((self.n_points,self.channels)).T
            except:
                raise Exception("数据异常！")
        return new_data_trans


    def _recv_fixed_len(self, n_bytes):
        b_data = b''
        flag_stop_recv = False
        b_count = 0
        while not flag_stop_recv:
            try:
                tmp_bytes = self.client_socket.recv(n_bytes - b_count)
            except socket.timeout:
                raise ValueError('No data is Getted.')

            if b_count == n_bytes or not tmp_bytes:
                flag_stop_recv = True

            b_count += len(tmp_bytes)
            b_data += tmp_bytes
        return b_data


    def run(self):
        count = 0
        flag = 0
        T = []
        # lock_read = threading.Lock()
        while True:
            if self.client_socket:
                # lock_read.acquire()
                try:
                    # t1 = time.perf_counter()
                    new_data = self.get_eye_data()
                    # t2 = time.perf_counter()
                    # T.append((t2-t1))
                    # print('眼动的发包速度：{0},平均时间：{1}'.format((t2-t1),np.mean(T)))
                except:
                    print('Some problems have arisen, can not receive eye data from socket.')
                    self.client_socket.close()
                    break
                    # lock_read.release()
                else:
                    event = new_data[2,:]
                    triggerPos = np.nonzero(event)[0]
                    # print('triggerPose is :{}, shape is {}'.format(triggerPos,triggerPos.shape[0]))
                    if triggerPos.shape[0] >= 1:
                        if flag == 0:
                            new_data[2,triggerPos[-1]] = 127
                            flag = 1
                        else:
                            flag = 0
                            new_data[2,triggerPos[-1]] = 255
                    self.update_buffer(new_data)

                    # print(new_data[2,:]/10000)
                    # if np.sum(new_data[2,:],axis=0) == 10000:
                    #     count += 1
                    #     print("Current trigger number: {}".format(count))
                    # print(new_data[:2,:]/10000)

                    # lock_read.release()
            

    def update_buffer(self, new_data):
        '''
        Update data buffer when a new package arrived,12 points
        '''
        n_points_buffer = self.n_points_buffer
        current_ptr = self.current_ptr
        # eventline = new_data[-1,:]
        # eye_triggerPos = np.nonzero(eventline)[0]
        self.data_buffer[:,np.mod(np.arange(current_ptr,current_ptr + self.n_points), n_points_buffer)] = new_data
        self.current_ptr = np.mod(current_ptr + self.n_points, n_points_buffer)
        self.nUpdata = self.nUpdata + self.n_points
        # if eye_triggerPos.shape[0] >= 1:
        #     print('{},{},{}'.format(eventline[eye_triggerPos],new_data.shape,self.current_ptr))

    
    def get_buffer_data(self):
        data_buffer = self.data_buffer
        current_ptr = self.current_ptr
        data = np.hstack([data_buffer[:, current_ptr:], data_buffer[:, :current_ptr]])
        evt_value_buff = data[2, :]
        return data,evt_value_buff


    def get_bufferNupdata(self):
        return self.nUpdata


    def set_bufferNupdata(self,nUpdata):
        self.nUpdata = nUpdata
    

    def reset_buffer(self):
        '''
        Reset data buffer.
        '''
        self.data_buffer = np.zeros((self.channels, self.n_points_buffer))  # data buffer
        self.current_ptr = 0
        self.nUpdate = 0
        self.read_data_flag = False

    def flag_receiver(self, value):
        self.read_data_flag = value

















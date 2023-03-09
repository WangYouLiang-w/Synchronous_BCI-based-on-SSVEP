import numpy as np
import numpy.matlib
import math
import scipy.io as sio
import warnings
from scipy import signal
from scipy import linalg as sLA
import mne
import readbdfdata


class PreProcessing():
    '''
    Adapted from Orion Han
    '''
    CHANNELS = [
        'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3',
        'F1','FZ','F2','F4','F6','F8','FT7','FC5',
        'FC3','FC1','FCZ','FC2','FC4','FC6','FC8','T7',
        'C5','C3','C1','CZ','C2','C4','C6','T8',
        'M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4',
        'CP6','TP8','M2','P7','P5','P3','P1','PZ',
        'P2','P4','P6','P8','PO7','PO5','PO3','POZ',
        'PO4','PO6','PO8','CB1','O1','OZ','O2','CB2'
    ] # M1: 33. M2: 43.

    def __init__(self,fs_down=1000, num_filter=1):
        '''
        init()
        :param filepath: .cnt or .mat of filepath
        :param t_begin:  start_time of pos(s)
        :param t_end:    end_time of pos(s)
        :param n_classes: nums of class
        :param fs_down: fs
        :param chans: select of Channels
        :param num_filter: nums of filter
        '''
        self.fs_down = fs_down
        self.num_filter = num_filter


    def resample_data(self, raw_data):
        '''
        :param raw_data: from method load_data.
        :return: raw_data_resampled, 4-D, numpy
            n_chans * n_samples * n_classes * n_trials
        '''

        if self.raw_fs > self.fs_down:
            raw_data_resampled = signal.resample(raw_data, round(self.fs_down*raw_data.shape[1]/self.raw_fs), axis=1)
        elif self.raw_fs < self.fs_down:
             warnings.warn('You are up-sampling, no recommended')
             raw_data_resampled = signal.resample(raw_data, round(self.fs_down*raw_data.shape[1]/self.raw_fs), axis=1)
        else:
            raw_data_resampled = raw_data

        return raw_data_resampled
    

    def _get_iir_sos_band(self, w_pass, w_stop):
        '''
        Get second-order sections (like 'ba') of Chebyshev type I filter.
        :param w_pass: list, 2 elements
        :param w_stop: list, 2 elements
        :return: sos_system
            i.e the filter coefficients.
        '''
        if len(w_pass) != 2 or len(w_stop) != 2:
            raise ValueError('w_pass and w_stop must be a list with 2 elements.')

        if w_pass[0] > w_stop[1] or w_stop[0] > w_stop[1]:
            raise ValueError('Element 1 must be greater than Element 0 for w_pass and w_stop.')
        
        if w_pass[0] < w_stop[0] or w_pass[1] > w_stop[1]:
            raise ValueError('It\'s a band-pass iir filter, please check the values between w_pass and w_stop.')
        
        wp = [2 * w_pass[0] / self.fs_down, 2 * w_pass[1] / self.fs_down]
        ws = [2 * w_stop[0] / self.fs_down, 2 * w_stop[1] / self.fs_down]

        if self.fs_down == 250:
            gpass = 3
            gstop = 40 #DB

        if self.fs_down == 1000:
            gpass = 3
            gstop = 10

        N, wn = signal.cheb1ord(wp, ws, gpass=gpass, gstop=gstop)
        f_b, f_a = signal.cheby1(N, rp=0.5, Wn=wn, btype='bandpass', output='ba')
        return f_b, f_a
    

    def filtered_data_iir(self, w_pass_2d, w_stop_2d, data):
        '''
        filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.
        :param w_pass_2d: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        :param w_stop_2d: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
        :param data: 2-d, numpy, from method load_data or resample_data.
            n_chans * n_samples
        :return: filtered_data: shape(nsub,nchans,nsamples)
        e.g.
        w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
        w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])
        '''
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')
        if self.num_filter > w_pass_2d.shape[1]:
            raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

        filtered_data = np.zeros((w_pass_2d.shape[1], data.shape[0], data.shape[1]))
        for idx_filter in range(self.num_filter):
            f_b,f_a = self._get_iir_sos_band(w_pass=[w_pass_2d[0, idx_filter], w_pass_2d[1, idx_filter]],
                                                                              w_stop=[w_stop_2d[0, idx_filter], w_stop_2d[1, idx_filter]])
            data1 = np.transpose(data,[1,0])
            flipdata = np.flip(data1, axis=0)
            currentdata = np.r_[flipdata,data1,flipdata]
            filter_data = signal.filtfilt(f_b, f_a, currentdata, axis=0, padtype = 'odd',padlen=3*(max(len(f_b), len(f_a))-1))[data1.shape[0]:2*data1.shape[0],:]
            filtered_data[idx_filter, :, :] = np.transpose(filter_data, [1,0])

        return filtered_data
    

    def fb_generate_cca_references(self, freqs, srate, T, Nsand, phases = None,n_harmonics: int = 1):
        '''
        generate_cca_references
        :param freqs shape(1,n)
        :param srate, int
        :param T, length of time, float
        :param Nsand, num of filter, int
        :param phases,shape(1,n)
        :param n_harmonics, int
        :return: Yf:shape(n, 2*n_harmonics, T*srate)
                 Pf:shape(n, T*srate, T*srate)
        '''
        Pf = np.zeros((Nsand, len(freqs),int(T*srate),int(T*srate)))
        if isinstance(freqs, int) or isinstance(freqs, float):
            freqs = [freqs] 
        freqs = np.array(freqs)[:, np.newaxis]
        if phases is None:
            phases = 0
        if isinstance(phases, int) or isinstance(phases, float):
            phases = [phases] 
        phases = np.array(phases)[:, np.newaxis]
        t = np.linspace(0, T, int(T*srate))
        
        for j in range(Nsand):
            Yf = []
            for i in range(j, n_harmonics):
                Yf.append(np.stack([
                    np.sin(2*np.pi*(i+1)*freqs*t + np.pi*phases*(i+1)),
                    np.cos(2*np.pi*(i+1)*freqs*t + np.pi*phases*(i+1))], axis=1))
            Yf = np.concatenate(Yf, axis=1)
            Y = np.transpose(Yf,[0,2,1])

            for i in range(len(freqs)):
                Q, _ = sLA.qr(Y[i, :, :], mode='economic')
                Pf[j, i, :, :] = Q @ Q.T  # (Np,Np)
        
        return Yf, Pf
    

    def generate_cca_references(self, freqs, srate, T, phases = None,n_harmonics: int = 1):
        '''
        generate_cca_references
        :param freqs shape(1,n)
        :param srate, int
        :param T, length of time, float
        :param phases,shape(1,n)
        :param n_harmonics, int
        :return: Yf:shape(n, 2*n_harmonics, T*srate)
                 Pf:shape(n, T*srate, T*srate)
        '''
        t = np.linspace(0, T, int(T*srate))
        Pf = np.zeros((len(freqs), int(T*srate), int(T*srate)))
        Yf = np.zeros((len(freqs), 2*n_harmonics, len(t)))
        x = np.zeros((2*n_harmonics,len(t)))
        for triggernum in range(len(freqs)):
            for i in range(n_harmonics):
                x[i*2,:] = np.sin(2*np.pi*(i+1)*freqs[triggernum]*t + np.pi*phases[triggernum]*(i+1))
                x[i*2+1,:] = np.cos(2*np.pi*(i+1)*freqs[triggernum]*t + np.pi*phases[triggernum]*(i+1))

            Q, _ = sLA.qr(x.T, mode='economic')
            # Pf[triggernum, :, :] = Q[:,:n_harmonics*2] @ Q[:, :n_harmonics*2].T 
            Pf[triggernum, :, :] = Q @ Q.T 
            Yf[triggernum, :, :] = x 

        return Yf, Pf

      

    def ShuffleSplit_mean(self, nTrail, n_splits):
        '''
        平均交叉验证
        :argument nTrail: num of Trial (int)
        :argument n_splits: n_splits-fold (int)
        :return:trainlice: shape(nTrail,n_splits)
                testslice: shape(nTrail,n_splits)
        '''
        myslice = np.zeros((nTrail, 1))
        test_num = int(nTrail/n_splits)
        testslice = np.zeros((nTrail, n_splits))
        trainlice = np.zeros((nTrail, n_splits))
        for i in range(n_splits):
            myslice[i*test_num:(i+1)*int(nTrail/n_splits)] = 1
            testslice[:,i] = myslice[:,0]
            trainlice[:,i] = 1 - myslice[:,0]
            myslice = np.zeros((nTrail, 1))

        return trainlice, testslice




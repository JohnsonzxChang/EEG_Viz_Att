import torch
import numpy as np
import scipy.io as mio
import torch.utils.data as DD
from scipy.signal import kaiserord, filtfilt, firwin, butter

def construct_filter(fs=250.0, cutoff_hz=[[6, 66]], type='kaiser'):
    if type == 'kaiser':
        width = 4.0 / fs
        ripple_db = 20.0
        N, beta = kaiserord(ripple_db, width)
        taps = []
        for i in range(len(cutoff_hz)):
            taps.append(firwin(N, [cutoff_hz[i][0], cutoff_hz[i][1]], window=('kaiser', beta), fs=fs, pass_zero="bandpass"))
        my_filter = lambda x: [filtfilt(taps[j], 1.0, x, axis=-1, padlen=x.shape[-1] - 2) for j in range(len(cutoff_hz))]
    elif type == 'firwin':
        taps = []
        for i in range(len(cutoff_hz)):
            taps.append(firwin(25, [cutoff_hz[i][0], cutoff_hz[i][1]], fs=fs, pass_zero="bandpass"))
        my_filter = lambda x: [filtfilt(taps[j], 1.0, x, axis=-1, padlen=x.shape[-1] - 2) for j in range(len(cutoff_hz))]
    elif type == 'butter':
        taps = []
        for i in range(len(cutoff_hz)):
            taps.append(butter(4, [cutoff_hz[i][0], cutoff_hz[i][1]], fs=fs, btype="bandpass"))
        my_filter = lambda x: [filtfilt(taps[j][0], taps[j][1], x, axis=-1, padlen=x.shape[-1] - 2) for j in range(len(cutoff_hz))]
    else:
        raise ValueError('type err')
    return my_filter

def load_data_refine(base_path: str, T: int, E: list, Mm=None, T0:int=125):
    if Mm is None:
        Mm = [53, 54, 55, 56, 57, 58, 59, 61, 62, 63]
    f0 = f'{base_path}/Freq_Phase.mat'
    Freq_Class = mio.loadmat(f0)['freqs'].squeeze()
    Phase_Class = mio.loadmat(f0)['phases'].squeeze()
    res = {}
    res["Freq_Class"] = Freq_Class
    res["Phase_Class"] = Phase_Class
    tmp = np.load(f'{base_path}/ssvep_dataset_data.npy')[E, :, :]
    tmp = tmp[:,Mm,:]
    lb = np.load(f'{base_path}/ssvep_dataset_label.npy')[E]

    # pre 125, valid 1250, post 1375, finish 1500, cut from 250 to 1250 (4s)
    print(f'loading data from {T0} to {T0+T}')
    tmp = tmp[:,:,T0:T0+T]
    res["Data"] = tmp
    res["Label"] = lb
    res["id"] = [e // 240 for e in E]
    assert max(res["id"]) <= 35
    return res

class base_torch_mix_dataset(DD.Dataset):
    def __init__(self, base_path:str, T0:int, E=None, with_freq=False, preprocess=False, multiplex=1,
                 filterbank=False, train=True, t_pre=None, return_id=False, with_index=False):
        super().__init__()
        self.T0 = T0
        self.multiplex = multiplex
        self.return_id = return_id
        self.base_path = base_path
        self.with_index = with_index
        EE_Train = []
        EE_Test = []
        if t_pre is None:
            t_pre = 125+125
        if E is None:
            raise ValueError('E is not given')
        else:
            if type(E) == list:
                for e in E:
                    EE_Train += list(range(e*6*40, e*6*40+5*40))
                    EE_Test += list(range(e*6*40+5*40, e*6*40+6*40))
            else:
                assert type(E) == int
                for e in range(E):
                    EE_Train += list(range(e*6*40, e*6*40+5*40))
                    EE_Test += list(range(e*6*40+5*40, e*6*40+6*40))
        # print(f'{EE_Train},,{EE_Test}')
        if train:
            self.E = EE_Train
        else:
            self.E = EE_Test
        self.with_freq = with_freq
        self.preprocess = preprocess
        if filterbank != False:
            print('construct filter bank')
            assert type(filterbank) == int
            filterbank = [[(i+1)*8-2,90] for i in range(filterbank)]
            self.filterbank = construct_filter(cutoff_hz=filterbank, type='butter')
        else:
            print('dont use filter bank')
            self.filterbank = None
        self._load_data(T0=t_pre)
        self._make_label()


    def _load_data(self, T0=125):
        print(f'cutting data into {self.multiplex} parts')
        assert int(self.T0*self.multiplex) <= 1250 - T0
        res = load_data_refine(self.base_path, int(self.T0*self.multiplex), self.E, T0=T0)
        self.data = res["Data"]
        if self.filterbank is not None:
            self.data = np.stack(self.filterbank(self.data), axis=1)
            assert self.data.shape[1] == 3
            print('finish filter...')
        self.freq = res["Label"]
        self.Freq_Class = res["Freq_Class"]
        self.id = np.array(res["id"])
        
    def _make_label(self):
        assert self.data.shape[0] == self.freq.shape[0]
        self.label = np.zeros_like(self.freq, dtype=int)
        for i in range(self.label.shape[0]):
            tmp = np.where(self.Freq_Class == self.freq[i])
            assert len(tmp) == 1
            self.label[i] = int(tmp[0])
        data = []
        label = []
        idid = []
        for i in range(self.multiplex):
            data.append(self.data[:,:,:,i*self.T0:(i+1)*self.T0])
            label.append(self.label)
            idid.append(self.id)
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)
        self.id = np.concatenate(idid, axis=0)

    def _preprocess_data(self, x):
        x = (x - np.mean(x, axis=-1, keepdims=True)
                     ) / np.std(x, axis=-1, keepdims=True)
        return x

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        data = self.data[item]
        id = self.id[item]
        if self.preprocess:
            data = self._preprocess_data(data)
        data = torch.tensor(data).type(torch.float)
        id = torch.tensor(id).type(torch.float)
        label = torch.tensor(self.label[item]).type(torch.LongTensor)
        if self.with_freq:
            freq = torch.tensor(self.freq[item]).type(torch.float)
            return data, freq, label
        else:
            if self.return_id:
                if self.with_index:
                    return data, id, label, item
                else:
                    return data, id, label
            else:
                return data, label


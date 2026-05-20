import os
import numpy as np
import pandas as pd
import glob
import re
import mne 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import sys 
from conf import BaseConfig
import random
from pycocotools.coco import COCO
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image

warnings.filterwarnings('ignore')

USE_SINGLE_LABEL = True
ROOT_PATH = os.environ.get('DATA_ROOT', os.getcwd())
DATASET_SS_PATH = f'{ROOT_PATH}/data/ss_bench'
UDF_SSMR_PATH = f'{ROOT_PATH}/data/udf_ssmr'
UDF_ROBOT_PATH = f'{ROOT_PATH}/data/udf_robot'  # configure as needed
UDF_VIZ_PATH = f'{ROOT_PATH}/data/udf_viz'
UDF_VIZ_OPENIMAGE_PATH = f'{ROOT_PATH}/data/zx-viz-sel-0203'
UDF_VIZ3d_PATH = f'{ROOT_PATH}/data/udf_viz3d'  # configure as needed

ALL_CLA = [
                "person", # removed: dominates ~64% of images, collapses training
                "bicycle", # (20, 35, 3501)
                "car", # (21, 35, 3501)
                "motorcycle", # (24, 35, 3501)
                "airplane", # (24, 35, 3501)
                "bus", # (22, 35, 3501)
                "train", # (19, 35, 3501)
                "truck", # (25, 35, 3501)
                "boat", # (25, 35, 3501)
                "traffic light", # (25, 35, 3501)
                "fire hydrant", # (22, 35, 3501)
                "stop sign", # (23, 35, 3501)
                "parking meter", # (21, 35, 3501)
                "bench", # (23, 35, 3501)
                "bird", # (25, 35, 3501)
                "cat", # (21, 35, 3501)
                "dog", # (24, 35, 3501)
                "horse", # (23, 35, 3501)
                "sheep", # (24, 35, 3501)
                "cow", # (21, 35, 3501)
                "elephant", # (22, 35, 3501)
                "bear", # (24, 35, 3501)
                "zebra", # (25, 35, 3501)
                "giraffe", # (21, 35, 3501)
                "backpack", # (22, 35, 3501)
                "umbrella", # (21, 35, 3501)
                "handbag", # (23, 35, 3501)
                "tie", # (24, 35, 3501)
                "suitcase", # (22, 35, 3501)
                "frisbee", # (22, 35, 3501)
                "skis", # (25, 35, 3501)
                "snowboard", # (24, 35, 3501)
                "sports ball", # (23, 35, 3501)
                "kite", # (22, 35, 3501)
                "baseball bat", # (22, 35, 3501)
                "baseball glove", # (20, 35, 3501)
                "skateboard", # (23, 35, 3501)
                "surfboard", # (24, 35, 3501)
                "tennis racket", # (21, 35, 3501)
                "bottle", # (21, 35, 3501)
                "wine glass", # (23, 35, 3501)
                "cup", # (22, 35, 3501)
                "fork", # (22, 35, 3501)
                "knife", # (19, 35, 3501)
                "spoon", # (25, 35, 3501)
                "bowl", # (21, 35, 3501)
                "banana", # (24, 35, 3501)
                "apple", # (23, 35, 3501)
                "sandwich", # (21, 35, 3501)
                "orange", # (25, 35, 3501)
                "broccoli", # (23, 35, 3501)
                "carrot", # (24, 35, 3501)
                "hot dog", # (22, 35, 3501)
                "pizza", # (23, 35, 3501)
                "donut", # (24, 35, 3501)
                "cake", # (23, 35, 3501)
                "chair", # (24, 35, 3501)
                "couch", # (24, 35, 3501)
                "potted plant", # (20, 35, 3501)
                "bed", # (20, 35, 3501)
                "dining table", # (23, 35, 3501)
                "toilet", # (21, 35, 3501)
                "tv", # (21, 35, 3501)
                "laptop", # (23, 35, 3501)
                "mouse", # (20, 35, 3501)
                "remote", # (25, 35, 3501)
                "keyboard", # (21, 35, 3501)
                "cell phone", # (20, 35, 3501)
                "microwave", # (17, 35, 3501)
                "oven", # (22, 35, 3501)
                "toaster", # (22, 35, 3501)
                "sink", # (23, 35, 3501)
                "refrigerator", # (24, 35, 3501)
                "book", # (21, 35, 3501)
                "clock", # (23, 35, 3501)
                "vase", # (22, 35, 3501)
                "scissors", # (16, 35, 3501)
                "teddy bear", # (23, 35, 3501)
                "hair drier", # (25, 35, 3501)
                "toothbrush", # (23, 35, 3501)
            ]


ALL_CLA_OPENIMAGE = [
    'Curtain',
 'French fries',
 'Surfboard',
 'Food',
 'Gas stove',
 'Goggles',
 'Saucer',
 'Bicycle',
 'Bottle',
 'House',
 'Cowboy hat',
 'Trousers',
 'Cart',
 'Dessert',
 'Wheel',
 'Animal',
 'Furniture',
 'Countertop',
 'Helmet',
 'Coffee cup',
 'Bicycle helmet',
 'Carnivore',
 'Beard',
 'Ice cream',
 'Drawer',
 'Car',
 'Clothing',
 'Coffee table',
 'Watercraft',
 'Miniskirt',
 'Canoe',
 'Truck',
 'Table',
 'Dairy Product',
 'Boat',
 'Fashion accessory',
 'Glasses',
 'Land vehicle',
 'Nightstand',
 'Plant',
 'Dog',
 'Jacket',
 'Vase',
 'Chair',
 'Building',
 'Coat',
 'Skateboard',
 'Handbag',
 'Houseplant',
 'Bed',
 'Camera',
 'Porch',
 'Fast food',
 'Computer keyboard',
 'Laptop',
 'Ladder',
 'Bicycle wheel',
 'Beer',
 'Auto part',
 'Flowerpot',
 'Horse',
 'Cheese',
 'Tie',
 'Sun hat',
 'Swimwear',
 'Fruit',
 'Home appliance',
 'Baked goods',
 'Door',
 'Vehicle',
 'Ball (Object)',
 'Tire',
 'Couch',
 'Rifle',
 'Fork',
 'Drink',
 'High heels',
 'Vegetable',
 'Tennis racket',
 'Weapon',
 'Footwear',
 'Motorcycle',
 'Vehicle registration plate',
 'Dress',
 'Computer monitor',
 'Shorts',
 'Ski',
 'Toy',
 'Van',
 'Cabinetry',
 'Skirt',
 'Brassiere',
 'Jeans',
 'Sunglasses',
 'Desk',
 'Flower',
 'Tableware',
 'Window',
 'Tea',
 'Coffee (drink)',
 'Spoon',
 'Hat',
 'Chest of drawers',
 'Fedora',
 'Sports equipment',
 'Shirt',
 'Wheelchair',
 'Suit',
 'Tree',
 'Paddle',
 'Kitchen & dining room table']


class Dataset_UDF_VIZ_Cla_singleclass(Dataset):
    def __init__(self, args:BaseConfig, seeds:int=None):
        super().__init__()

        if seeds is not None:
            np.random.seed(seeds)

        self.args = args
        # assert flag in ['trn', 'val', 'tst'], f"Flag must be one of ['trn', 'val', 'tst'], got {flag}"
        assert os.path.exists(UDF_VIZ_PATH), f"Dataset path {UDF_VIZ_PATH} does not exist"
        # all_files = [os.path.join(UDF_VIZ_PATH, 'zx-1122', 'erp.fif'), os.path.join(UDF_VIZ_PATH, 'zfn-1125', 'erp.fif')]
        # file_balence = [20, 16]
        # all_files = [os.path.join(UDF_VIZ_PATH, 'zfn-erp-1128', 'erp.fif')]
        # file_balence = [200]
        # file_balence = [36]
        all_files = [os.path.join(UDF_VIZ_OPENIMAGE_PATH, 'zxc-0203', 'sub_erp2.fif')]
        file_balence = [180]
        # file_balence = [20]
        # all_files = [os.path.join(UDF_VIZ_PATH, 'zfn-1125', 'erp.fif')]
        # file_balence = [16]
        assert len(all_files) > 0, f"No .fif files found in {UDF_VIZ_PATH}"
        self.all_data = []
        self.all_regs = []
        self.all_img = []
        self.all_subjects = []

        for i, file in enumerate(all_files):
            epochs = mne.read_epochs(file, preload=True, verbose=False) # trials x chns x time (chns=32 + 3)
            # print epoch tmin and tmax within 0 as fixation onset
            print(f"Loaded {file} with {len(epochs)} epochs, tmin={epochs.tmin}, tmax={epochs.tmax}")
            dt, lb, ig = [], [], []
            cnt = 0
            for ll, tgt in enumerate(ALL_CLA_OPENIMAGE):
            # for ll, tgt in enumerate([
            #     # 'accessory',
            #     'animal',
            #     # 'appliance',
            #     # 'electronic',
            #     'food',
            #     # 'furniture',
            #     # 'indoor',
            #     # 'kitchen',
            #     # 'outdoor',
            #     #  'person',
            #     'sports',
            #     'vehicle'
            # ]):
                ep = epochs[tgt]
                code_to_name = {code: name for name, code in ep.event_id.items()}

                event_codes = ep.events[:, 2]  # (n_epochs,)
                img_id = [str(code_to_name[c].split('/')[-1].split('.')[0]) for c in event_codes]
                # img_id = 126050  # 你的图片 id（int）
                # 1) 找到这张图片对应的所有标注 id
                all_lblb = np.ones((len(img_id))) * ll
                all_img = np.array(img_id)
                tmp = ep.get_data()*1e6
                assert all_lblb.shape[0] == tmp.shape[0] == all_img.shape[0], f'{all_lblb.shape}...{tmp.shape}...{all_img.shape}'
                if tmp.shape[0] >= file_balence[i]:
                    all_sel = np.random.choice(list(range(tmp.shape[0])), size=file_balence[i], replace=False)
                    dt.append(tmp[all_sel,:,:]) # # 89, 83, 17
                    lb.append(all_lblb[all_sel]) # 89, 83, 17
                    ig.append(all_img[all_sel])
                else:
                    print(f'data class imbalance in class {tgt} with {tmp.shape[0]} shorter than {file_balence[i]}')
                    dt.append(tmp) # # 89, 83, 17
                    lb.append(all_lblb) # 89, 83, 17
                    ig.append(all_img)
                # dt.append(tmp)
                # lb.append(np.array([cnt]*tmp.shape[0]))
                cnt += 1
            dt = np.vstack(dt) # trials x chns x time
            lb = np.concatenate(lb, axis=0) # trials
            ig = np.concatenate(ig, axis=0) # trials
            print(dt.shape, lb.shape, ig.shape) # (420, 35, 1001) (420,)

            regVal = lb  # 0,1,2,3,4,5
            data = dt[:, self.args.chn_sel, :] # only use first chn_num channels 5501 = 1000 void + 4000 data + 501 void
            # regVal = epochs[:, 32+1:32+3, :] # y,z position

            # data = data[:,:,1200:3700]

            for j in range(1): # 4000 / 500 = 8 * 2 = 16 mux
                t_len = self.args.t_len
                # mid = 750 + 250 + j*(t_len//2)
                mid = self.args.t0 + t_len//2 + j*(t_len)
                start = mid - (t_len//2); end = mid + (t_len//2)
                print(f'start: {start}...end:{end}')
                self.all_data.append(data[:, :, start:end].astype(np.float32))
                self.all_regs.append(regVal.astype(np.int32)) # use the last time point as regression target
                self.all_subjects.append(np.ones((data.shape[0],), dtype=int) * i)
                self.all_img.append(ig)
        self.all_data = np.concatenate(self.all_data, axis=0)
        self.all_regs = np.concatenate(self.all_regs, axis=0)
        self.all_subjects = np.concatenate(self.all_subjects, axis=0)
        self.all_img = np.concatenate(self.all_img, axis=0)

        self.trn_sel = list(range(self.all_data.shape[0]))

        self.val_sel = np.random.choice(self.trn_sel, size=int(0.2*len(self.trn_sel)), replace=False)
        self.trn_sel = list(set(self.trn_sel) - set(self.val_sel))

        # 基于训练集计算 pos_weight (neg_count / pos_count per class)
        # Single label counting for optionally re-weighting classes later (not heavily needed for standard CE if balanced)
        # if len(trn_labels.shape) == 1:
        #     pos_count = np.bincount(trn_labels, minlength=len(ALL_CLA))
        # else:
        #     pos_count = trn_labels.sum(axis=0)         # (80,)
        # neg_count = trn_labels.shape[0] - pos_count
        # self.pos_weight = np.where(pos_count > 0, neg_count / pos_count, 1.0).astype(np.float32)
        # print(f"Total samples: {self.all_data.shape[0]}, Training: {len(self.trn_sel)}, Validation: {len(self.val_sel)}")
        # print(f"pos_weight range: [{self.pos_weight.min():.2f}, {self.pos_weight.max():.2f}], person={self.pos_weight[0]:.2f}")
        self.flag = None 

    def get_flag(self, flag:str):
        assert flag in ['trn', 'val', 'tst'], f"Flag must be one of ['trn', 'val', 'tst'], got {flag}"
        if flag == 'trn':
            self.flag = self.trn_sel
        elif flag == 'val':
            self.flag = self.val_sel
        else:
            raise ValueError(f"Flag must be 'trn' or 'val', got {flag}")
        print(f'Validating {self.val_sel[:10]} ... samples for {flag} set.')
        self.all_data = self.all_data[self.flag]
        self.all_regs = self.all_regs[self.flag]
        self.all_subjects = self.all_subjects[self.flag]
        self.all_img = self.all_img[self.flag]

    
    def __len__(self):
        return self.all_data.shape[0]
    
    def _get_img(self, id):
        img_root = getattr(self.args, 'coco_img_path', None)
        if img_root is None:
            if 'win' in sys.platform:
                img_root = None  # <SET YOUR DATA PATH>
            else:
                img_root = None  # set via config.coco_img_path
        id = str(id).zfill(12)
        path = os.path.join(img_root, id + ".jpg")
        if os.path.isfile(path):
            return Image.open(path).convert("RGB")
        else:
            raise FileNotFoundError(f'{path} not found')
    
    def __getitem__(self, idx):
        # data = self.all_data[self.flag]
        # regs = self.all_regs[self.flag]
        # subjects = self.all_subjects[self.flag]
        # return torch.tensor(self.all_data[idx]), torch.tensor(self.all_regs[idx]), [torch.tensor(self.all_subjects[idx], dtype=torch.long), torch.tensor(self.all_img[idx],dtype=torch.long)]
        return torch.tensor(self.all_data[idx]), torch.tensor(self.all_regs[idx], dtype=torch.long), torch.tensor(self.all_subjects[idx], dtype=torch.long)


def non_linear_transfer(x, kind='exp'):
    if kind == 'exp':
        return np.sign(x) * (np.exp(np.abs(x)) - 1)
    elif kind == 'log':
        return np.sign(x) * np.log(np.abs(x) + 1)
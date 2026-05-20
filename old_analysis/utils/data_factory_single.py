from .data_loader_single import *
from .data_loader_coco_erp_vi_pair import *
from conf import BaseConfig
from torch.utils.data import Dataset, DataLoader

data_dict = {
    'udf_viz_single': Dataset_UDF_VIZ_Cla_singleclass,
}
FLAGS = ['trn', 'val', 'tst']
def get_data_loader(args:BaseConfig, flag=None):
    assert flag in FLAGS, f"Flag must be one of {FLAGS}, got {flag}"
    if flag is None:
        flag = 'val'
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'tst') else True
    drop_last = False
    batch_size = args.batch_size
    workers = args.num_workers
    pin_momory = args.pin_memory

    drop_last = False
    data_set = Data(
        args = args,
        flag=flag,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=workers,
        pin_memory=pin_momory,
        drop_last=drop_last)
    return data_set, data_loader



def get_data_loader_cutt(args:BaseConfig):
    Data = data_dict[args.data]
    drop_last = False
    batch_size = args.batch_size
    workers = args.num_workers
    pin_momory = args.pin_memory

    drop_last = False
    data_set_trn = Data(
        args = args,
        seeds = args.seed,
    )
    data_set_trn.get_flag('trn')
    data_set_val = Data(
        args = args,
        seeds = args.seed,
    )
    data_set_val.get_flag('val')

    print(len(data_set_trn), len(data_set_val))
    data_loader_trn = DataLoader(
        data_set_trn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_momory,
        drop_last=drop_last)
    data_loader_val = DataLoader(
        data_set_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_momory,
        drop_last=drop_last)
    return data_set_trn, data_loader_trn, data_set_val, data_loader_val
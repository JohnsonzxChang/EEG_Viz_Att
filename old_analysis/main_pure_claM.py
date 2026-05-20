from conf import BaseConfigViz
from tasks import Exp_ClassificationM
import torch
import numpy as np

import argparse
from datetime import datetime

def main():
    # arg = argparse.ArgumentParser(description='VIZ T0 Scan')
    # arg.add_argument('--t0', type=int, default=0, help='T0 offset from 1000ms')
    # args = arg.parse_args()
    # T0 = args.t0 if hasattr(args, 't0') else 0
    # T0 = int(T0)
    # T0 = 1000 + T0 # 1000 + 0

    T0 = 1000 # + T0 # 1000 + 0
    # config = BaseConfigVizCoCo()
    # config.t0 = T0
    # config.comment = f'{datetime.now()}-zx-vizCoCocla-noclip{T0}-{config.model}'

    
    config = BaseConfigViz()
    config.t0 = T0
    config.comment = f'{datetime.now()}-zx-vizCoCocla{T0}-{config.model}'

    # config = BaseConfigEEGEMGReg()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Setting t0 to {config.t0}')
    # exp = Exp_ClassificationClip(config)
    exp = Exp_ClassificationM(config)
    exp.train('aaaa')

if __name__ == '__main__':
    main()

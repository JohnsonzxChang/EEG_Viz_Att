import torch
import numpy as np
from datetime import datetime
from conf import BaseConfigVizCoCo
from tasks import Exp_Retrieval


def main():
    config = BaseConfigVizCoCo()
    config.t0 = 500
    config.model = 'ATM'
    config.comment = f'{datetime.now()}-retrieval-{config.model}'

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    exp = Exp_Retrieval(config)
    exp.train('retrieval_run')


if __name__ == '__main__':
    main()

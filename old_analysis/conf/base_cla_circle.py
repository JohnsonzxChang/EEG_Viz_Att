from .base_cla_viz import BaseConfigViz
from datetime import datetime


class BaseConfigCircle(BaseConfigViz):
    """Multi-label classification: ASL + Circle Loss hybrid.

    Adds the following hyper-parameters on top of BaseConfigViz:

    feat_dim       : encoder bottleneck dimension (enables feat_head in CNN/TF).
                     Circle Loss is computed on these feat_dim-dimensional embeddings.
    circle_gamma   : Circle Loss scale factor γ.
    circle_margin  : Circle Loss margin m  (Δ_p = 1-m, Δ_n = m).
    circle_lambda  : weight of Circle Loss  →  L = L_ASL + λ · L_circle.
    circle_jaccard : if True, weight positive pairs by Jaccard label similarity.
    """

    def __init__(self):
        super().__init__()
        # task stays 'classification', data stays 'udf_viz_m'

        # Encoder bottleneck — activates the feat_head → cls_head split
        self.feat_dim = 256

        # Circle Loss hyper-parameters
        self.circle_gamma   = 64
        self.circle_margin  = 0.25
        self.circle_lambda  = 0.1    # start small; tune in [0.05, 0.5]
        self.circle_jaccard = True

        self.loggerdir = './logs_Visual'
        self.comment   = f'{datetime.now()}-circle-{self.model}'

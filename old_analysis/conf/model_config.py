class ModelConfig:
    def __init__(self):
        # Transformer Encoder
        self.n_heads = 8
        self.d_ff = 2048
        self.dropout = 0.1
        self.e_layers = 6

        # STDP
        self.a_pos = 0.01
        self.a_neg = 0.01
        self.tau_pos = 20.0
        self.tau_neg = 20.0

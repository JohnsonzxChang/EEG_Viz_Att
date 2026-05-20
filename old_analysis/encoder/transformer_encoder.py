import torch
import torch.nn as nn

class TFEncoder(nn.Module):
    def __init__(self, args):
        super(TFEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.enc_in, nhead=args.n_heads, dim_feedforward=args.d_ff, dropout=args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.e_layers)
        d_feat = int(getattr(args, "eeg_feat_dim", getattr(args, "d_model", 256)))
        self.feat_head = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(args.enc_in * args.t_len, d_feat),
        )
        self.cls_head = nn.Linear(d_feat, args.num_classes)

    def forward_features(self, x, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        # x: [Batch, Channel, Input length]
        x = x.permute(2, 0, 1)
        output = self.transformer_encoder(x, src_key_padding_mask=None)
        output = output.permute(1, 0, 2)
        output = output.reshape(output.size(0), -1)
        feat = self.feat_head(output)
        return feat

    def forward(self, x, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        feat = self.forward_features(x, padding_mask=padding_mask, enc_self_mask=enc_self_mask, dec_self_mask=dec_self_mask)
        return self.cls_head(feat)

    def forward_all(self, x, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        feat = self.forward_features(x, padding_mask=padding_mask, enc_self_mask=enc_self_mask, dec_self_mask=dec_self_mask)
        logits = self.cls_head(feat)
        return feat, logits

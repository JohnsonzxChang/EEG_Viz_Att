import torch
import torch.nn as nn
import math 

class CNN_Encoder(nn.Module):
    def __init__(self, args):
        super(CNN_Encoder, self).__init__()
        self.conv1 = nn.Conv1d(args.enc_in, 64, kernel_size=13, stride=4, padding=6)
        self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)  
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)  
        self.dropout = nn.Dropout(args.dropout)
        if hasattr(args, "feat_dim"):
            d_feat = args.feat_dim
            self.feat_head = nn.Linear(128 * int(math.ceil(math.ceil((args.t_len // 4) / 2) / 2)) , d_feat)
            print(f'feature dim : {int(math.ceil(math.ceil(args.t_len // 4 / 2)) / 2)} to {d_feat}')
            self.cls_head = nn.Linear(d_feat, args.num_classes)
        else:
            self.feat_head = nn.Linear(128 * int(math.ceil(math.ceil((args.t_len // 4) / 2) / 2)) , args.num_classes)
            self.cls_head = nn.Linear(args.num_classes, args.num_classes)
    

    def forward_features(self, x, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        out = self.conv1(x)
        out = self.relu1(out)
        # print(out.shape)
        # out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        # print(out.shape)
        # out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        # print(out.shape)
        # out = self.pool3(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        feat = self.feat_head(out)
        return feat

    def forward(self, x, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        feat = self.forward_features(x, padding_mask=padding_mask, enc_self_mask=enc_self_mask, dec_self_mask=dec_self_mask)
        return self.cls_head(feat)

    def forward_all(self, x, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        feat = self.forward_features(x, padding_mask=padding_mask, enc_self_mask=enc_self_mask, dec_self_mask=dec_self_mask)
        logits = self.cls_head(feat)
        return feat, logits
    

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.enc_in = 33
            self.t_len = 500
            self.num_classes = 80
            self.feat_dim = 256
    args = Args()
    model = CNN_Encoder(args)
    x = torch.randn(16, 33, 500)  # Batch size of 16, 33 channels, sequence length of 500
    feat = model(x)
    print("Feature shape:", feat.shape)

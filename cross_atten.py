
import torch.nn as nn

# from transforms import *
from torch.nn.init import normal_ as normal
from torch.nn.init import constant_ as constant

from typing import Optional
from torch import Tensor
import random
import math
import torch

#from ctcdecode.third_party.boost_1_67_0.libs.numeric.odeint.performance.plot_result import width


class BEF(nn.Module):
    def __init__(self, channel, reduction=8):
        super(BEF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputt):
        x = inputt.permute(0, 2, 1).contiguous()
        b, c, f = x.size()
        gap = self.avg_pool(x).view(b, c)
        y = self.fc(gap).view(b, c, 1)
        out = x * y.expand_as(x)

        return out.permute(0, 2, 1).contiguous()


class SA(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(SA, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # MLP, used for FFN
        self.activation = nn.ReLU(inplace=True)
        self.linear_in = nn.Linear(d_model, dim_feedforward)
        self.dropout_mlp = nn.Dropout(dropout)
        self.linear_out = nn.Linear(dim_feedforward, d_model)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)



    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                val=None):
        src_self = self.self_attention(src, src, value=val if val is not None else src,
                                       attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = src + self.drop1(src_self)
        src = self.norm1(src)
        tmp = self.linear_out(self.dropout_mlp(self.activation(self.linear_in(src))))

        src = self.norm2(src + self.drop2(tmp))

        return src


class CA(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CA, self).__init__()
        self.crs_attention1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.crs_attention2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP, used for FF
        self.activation = nn.ReLU(inplace=True)
        self.linear_in_1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_mlp_1 = nn.Dropout(dropout)
        self.linear_out_1 = nn.Linear(dim_feedforward, d_model)
        self.drop_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        self.linear_in_2 = nn.Linear(d_model, dim_feedforward)
        self.dropout_mlp_2 = nn.Dropout(dropout)
        self.linear_out_2 = nn.Linear(dim_feedforward, d_model)
        self.drop_2 = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model)



    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                ):
        src1_cross = self.crs_attention1(query=src1,
                                         key=src2,
                                         value=src2, attn_mask=src2_mask,
                                         key_padding_mask=src2_key_padding_mask)[0]

        src2_cross = self.crs_attention2(query=src2,
                                         key=src1,
                                         value=src1, attn_mask=src1_mask,
                                         key_padding_mask=src1_key_padding_mask)[0]

        src1 = src1 + self.drop1(src1_cross)
        src1 = self.norm1(src1)
        tmp = self.linear_out_1(self.dropout_mlp_1(self.activation(self.linear_in_1(src1))))  # FFN


        src1 = self.norm_1(src1 + self.drop_1(tmp))

        src2 = src2 + self.drop2(src2_cross)
        src2 = self.norm2(src2)
        tmp = self.linear_out_2(self.dropout_mlp_2(self.activation(self.linear_in_2(src2))))  # FFN


        src2 = self.norm_2(src2 + self.drop_2(tmp))

        return src1, src2




if __name__ == "__main__":
    model = SA(128,8)
    input_tensor = torch.rand(1, 224, 128)

    # 将模型设为 eval 模式
    model.eval()

    # 前向传播
    with torch.no_grad():
        x = model(input_tensor)

    # 打印每个 pyramid 特征的 shape
    # print("P2 shape:", P2.shape)  # [B, 256, T, H/4, W/4]  或者根据上采样后尺寸
    # print("P3 shape:", P3.shape)  # [B, 256, T, H/8, W/8]
    # print("P4 shape:", P4.shape)  # [B, 256, T, H/16, W/16]
    # print("P5 shape:", P5.shape)  # [B, 256, T, H/32, W/32]
    print("P_out shape:", x.shape)
import torch
import torch.nn as nn



class glff(nn.Module):
    def __init__(self, dim, num_heads=8, group_split=[8], kernel_sizes=[9],
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes)  == len(group_split)

        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5

        self.kernel_sizes = kernel_sizes
        #self.window_size = window_size
        self.group_split = group_split

        convs = []
        act_blocks = []
        qkvs = []
        for i, k in enumerate(kernel_sizes):
            heads = group_split[i]
            if heads == 0:
                continue

            convs.append(
                nn.Conv3d(
                    3 * self.dim_head * heads,
                    3 * self.dim_head * heads,
                    kernel_size=(1,k,k),padding=(0,k//2,k//2),


                    groups=3 * self.dim_head * heads
                )
            )

            qkvs.append(nn.Conv3d(dim, 3 * heads * self.dim_head, 1, 1, 0, bias=qkv_bias))



        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)

        self.proj = nn.Conv3d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)


    def high_fre_attntion(self, x, to_qkv, mixer, attn_block):
        B, C, D, H, W = x.shape

        qkv = to_qkv(x)                     # (B, 3*m*d, D, H, W)
        qkv = mixer(qkv)                    # DWConv3d

        qkv = qkv.reshape(B, 3, -1, D, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q * k) * self.scalor

        attn = torch.softmax(attn,dim=1)
        attn = self.attn_drop(attn)

        out = attn * v
        return out

    def forward(self, x):
        res = []

        # Local
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))

        out = torch.cat(res, dim=1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


if __name__ == "__main__":
    model = glff(256, 8, group_split=[8, 0], kernel_sizes=[5])
    x = torch.rand(1, 256, 8, 56, 56)
    out = model(x)
    print(out.shape)

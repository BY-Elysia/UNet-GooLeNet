import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
class Residual2(nn.Module):
    def __init__(self, cross_fn,fn1,fn2,):
        super().__init__()
        self.cross_fn = cross_fn
        self.fn1 = fn1
        self.fn2 = fn2
    def forward(self, x, x2, **kwargs):
        # return self.fn(x, x2, **kwargs) + x
        x_att = self.fn1(x)
        x2_att = self.fn2(x2)
        x_cross = self.cross_fn(x,x2,**kwargs)
        x2_cross = self.cross_fn(x2, x, **kwargs)
        return x_att+x_cross+x,x2_att+x2_cross+x2

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class PreNorm2(nn.Module):
    def __init__(self, dim, cross_fn,fn1,fn2,):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cross_fn = cross_fn
        self.fn1 = fn1
        self.fn2 = fn2
    def forward(self, x, x2, **kwargs):

        x = self.norm(x)
        x2 = self.norm(x2)
        x_att = self.fn1(x)
        x2_att = self.fn2(x2)
        return self.fn(self.norm(x), self.norm(x2), **kwargs)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
            # attn = dots
            # vis_tmp(dots)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)
        return out
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
class Multi_Attention(nn.Module):
    def __init__(self,dim, heads, dim_head, mlp_dim, dropout,softmax=True):
        super().__init__()
        self.attention1 = Attention(dim, heads=heads, dim_head=dim_head, dropout=0)
        self.attention2 = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.cross_attention_cl = Cross_Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout,
                        softmax=softmax)
        self.cross_attention_seg = Cross_Attention(dim, heads=heads,
                                               dim_head=dim_head, dropout=dropout,
                                               softmax=softmax)
        self.x_att_norm = nn.LayerNorm(dim)
        self.m_att_norm = nn.LayerNorm(dim)
        self.x_mlp_norm = nn.LayerNorm(dim)
        self.m_mlp_norm = nn.LayerNorm(dim)
        self.x_feed = FeedForward(dim, mlp_dim, dropout = dropout)
        self.m_feed = FeedForward(dim, mlp_dim, dropout=dropout)
    def forward(self, x, m, mask = None):

        x_norm = self.x_att_norm(x)
        m_norm = self.m_att_norm(m)
        x_att = self.attention1(x_norm)
        m_att = self.attention2(m_norm)


        x_cross = self.cross_attention_cl(x_norm, m_norm)
        m_cross = self.cross_attention_cl(m_norm, x_norm)
        x_mlp_in,m_mlp_in = x_att + x_cross + x, m_att + m_cross + m
        # x_mlp_in, m_mlp_in =  x_cross + x,  m_cross + m

        x_mlp_norm = self.x_mlp_norm(x_mlp_in)
        m_mlp_norm = self.m_mlp_norm(m_mlp_in)
        x_feed = self.x_feed(x_mlp_norm)
        m_feed = self.m_feed(m_mlp_norm)

        return  x_mlp_in+x_feed,m_mlp_in+m_feed



class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, decoder_pos_size,softmax=True):
        super().__init__()
        self.conv_cl = Conv2dReLU(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv_seg = Conv2dReLU(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(Multi_Attention(dim, heads = heads,dim_head = dim_head,mlp_dim=mlp_dim, dropout = dropout,softmax=softmax))
        # self.att_cl = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        # self.att_seg = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        # self.cross_att_1 = Cross_Attention(dim, heads = heads,dim_head = dim_head, dropout = dropout,softmax=softmax)
        self.pos_embedding_decoder_cl = nn.Parameter(torch.zeros(1, dim,
                                                              decoder_pos_size,
                                                              decoder_pos_size))
        self.pos_embedding_decoder_seg = nn.Parameter(torch.zeros(1, dim,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
    def forward(self, x, m, mask = None):
        b,c,h,w = x.shape
        x = self.conv_cl(x)
        m = self.conv_seg(m)
        x = x+self.pos_embedding_decoder_cl
        m = m + self.pos_embedding_decoder_seg
        x = rearrange(x, 'b c h w -> b (h w) c')
        m = rearrange(m, 'b c h w -> b (h w) c')
        """target(query), memory"""
        for attn in self.layers:
            x,m = attn(x, m, mask = mask)
        x = rearrange(x, 'b (h w) c -> b c h w',h=h)
        m = rearrange(m, 'b (h w) c -> b c h w',h=h)
        return x,m


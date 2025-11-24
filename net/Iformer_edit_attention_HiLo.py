import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/data2/lhq/PairLIE_edit')
from pdb import set_trace as stx
import numbers
from einops import rearrange
import sys
sys.path.append('/home/code/unsupervised-light-enhance-ICLR2025-main')
from dct_util import dct_2d,idct_2d,low_pass,low_pass_and_shuffle,high_pass
from thop import profile
import time

operation_seed_counter = 0
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

class AugmentNoise(object):
    def __init__(self, style):
        #print(style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# 修改轻量级高低频注意力机制
class HiLoAttention(nn.Module):
    """
    HiLo Attention

    Paper: Fast Vision Transformers with HiLo Attention
    Link: https://arxiv.org/abs/2205.13213
    """
    def __init__(self, dim, num_heads=1, bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads) # 每个注意力头的通道数
        self.dim = dim

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)  # 根据alpha来确定分配给低频注意力的注意力头的数量
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim   # 确定低频注意力的通道数

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads # 总的注意力头个数-低频注意力头的个数==高频注意力头的个数
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim  # 确定高频注意力的通道数, 总通道数-低频注意力通道数==高频注意力通道数

        # local window size. The `s` in our paper.
        self.ws = window_size  # 窗口的尺寸, 如果ws==2, 那么这个窗口就包含4个patch(或token)

        # 如果窗口的尺寸等于1,这就相当于标准的自注意力机制了, 不存在窗口注意力了; 因此,也就没有高频的操作了,只剩下低频注意力机制了
        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # Low frequence attention (Lo-Fi)
        # 如果低频注意力头的个数大于0, 那就说明存在低频注意力机制。 然后,如果窗口尺寸不为1, 那么应当为每一个窗口应用平均池化操作获得低频信息,这样有助于降低低频注意力机制的计算复杂度 （如果窗口尺寸为1,那么池化层就没有意义了）
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # High frequence attention (Hi-Fi)
        # 如果高频注意力头的个数大于0, 那就说明存在高频注意力机制
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    # 高频注意力机制
    def hifi(self, x):
        B, H, W, C = x.shape

        # 每行有w_group个窗口, 每列有h_group个窗口;
        h_group, w_group = H // self.ws, W // self.ws

        # 总共有total_groups个窗口; 例如：HW=14*14=196个patch; 窗口尺寸为ws=2表示:每行每列有2个patch; 总共有:(14/2)*(14/2)=49个窗口,每个窗口有2*2=4个patch
        total_groups = h_group * w_group

        #通过reshape操作重塑X: (B,H,W,C) --> (B,h_group,ws,w_group,ws,C) --> (B,h_group,w_group,ws,ws,C)   H=h_group*ws, W=w_group*ws
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        # 通过线性层生成qkv: (B,h_group,w_group,ws,ws,C) --> (B,h_group,w_group,ws,ws,3*h_dim) --> (B,total_groups,ws*ws,3,h_heads,head_dim) -->(3,B,total_groups,h_heads,ws*ws,head_dim)    h_dim=h_heads*head_dim
        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        # q:(B,total_groups,h_heads,ws*ws,head_dim); k:(B,total_groups,h_heads,ws*ws,head_dim); v:(B,total_groups,h_heads,ws*ws,head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 在每个窗口内计算: 所有patch pairs之间的注意力得分: (B,total_groups,h_heads,ws*ws,head_dim) @ (B,total_groups,h_heads,head_dim,ws*ws) = (B,total_groups,h_heads,ws*ws,ws*ws);  ws*ws:表示一个窗口内的patch的数量
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 通过注意力矩阵对Value矩阵进行加权: (B,total_groups,h_heads,ws*ws,ws*ws) @ (B,total_groups,h_heads,ws*ws,head_dim) = (B,total_groups,h_heads,ws*ws,head_dim) --transpose->(B,total_groups,ws*ws,h_heads,head_dim)--reshape-> (B,h_group,w_group,ws,ws,h_dim) ;    h_dim=h_heads*head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)

        # 通过reshape操作重塑, 恢复与输入相同的shape: (B,h_group,w_group,ws,ws,h_dim) --transpose-> (B,h_group,ws,w_group,ws,h_dim) --reshape-> (B,h_group*ws,w_group*ws,h_dim) ==(B,H,W,h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
        # 通过映射层进行输出: (B,H,W,h_dim)--> (B,H,W,h_dim)
        x = self.h_proj(x)
        return x

    # 低频注意力机制
    def lofi(self, x):
        B, H, W, C = x.shape
        # 低频注意力机制中的query来自原始输入x: (B,H,W,C) --> (B,H,W,l_dim) --> (B,HW,l_heads,head_dim) -->(B,l_heads,HW,head_dim);   l_dim=l_heads*head_dim;
        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        # 如果窗口尺寸大于1, 在每个窗口执行池化 (如果窗口尺寸等于1,没有池化的必要)
        if self.ws > 1:
            # 重塑维度以便进行池化操作:(B,H,W,C) --> (B,C,H,W)
            x_ = x.permute(0, 3, 1, 2)
            # 在每个窗口执行池化操作: (B,C,H,W) --sr-> (B,C,H/ws,W/ws) --reshape-> (B,C,HW/(ws^2)) --permute-> (B, HW/(ws^2), C);   HW=patch的总数, 每个池化窗口内有: (ws^2)个patch, 池化完还剩下：HW/(ws^2)个patch; 例如：HW=196个patch,每个池化窗口有(2^2=4)个patch,池化完还剩下49个patch【每个patch汇总了之前4个patch的信息】
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # 将池化后的输出通过线性层生成kv:(B,HW/(ws^2),C) --l_kv-> (B,HW/(ws^2),l_dim*2) --reshape-> (B,HW/(ws^2),2,l_heads,head_dim) --permute-> (2,B,l_heads,HW/(ws^2),head_dim)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            # 如果窗口尺寸等于1, 那么kv和q一样, 来源于原始输入x: (B,H,W,C) --l_kv-> (B,H,W,l_dim*2) --reshape-> (B,HW,2,l_heads,head_dim) --permute-> (2,B,l_heads,HW,head_dim);  【注意: 如果窗口尺寸为1,那就不会执行池化操作,所以patch的数量也不会减少,依然是HW个patch】
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)

        # 以ws>1为例: k:(B,l_heads,HW/(ws^2),head_dim);  v:(B,l_heads,HW/(ws^2),head_dim)
        k, v = kv[0], kv[1]

        # 计算q和k之间的注意力矩阵: (B,l_heads,HW,head_dim) @ (B,l_heads,head_dim,HW/(ws^2)) == (B,l_heads,HW,HW/(ws^2))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 通过注意力矩阵对Value矩阵进行加权: (B,l_heads,HW,HW/(ws^2)) @ (B,l_heads,HW/(ws^2),head_dim) == (B,l_heads,HW,head_dim) --transpose->(B,HW,l_heads,head_dim)--reshape-> (B,H,W,l_dim);   l_dim=l_heads*head_dim
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        # 通过映射层输出: (B,H,W,l_dim)-->(B,H,W,l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        # B, N, C = x.shape
        # # H = W = 每一列/行有多少个patch
        # H = W = int(N ** 0.5)
        # 将X重塑为四维: (B,N,C) --> (B,H,W,C)   【注意: 这里的H/W并不是图像的高和宽】
        x = x.reshape(B, H, W, C)

        # 如果分配给高频注意力的注意力头的个数为0,那么仅仅执行低频注意力
        if self.h_heads == 0:
            # (B,H,W,C) --> (B,H,W,l_dim)  此时,C=l_dim,因为所有的注意力头都分配给了低频注意力
            x = self.lofi(x)
            return x.reshape(B, N, C)

        # 如果分配给低频注意力的注意力头的个数为0,那么仅仅执行高频注意力
        if self.l_heads == 0:
            # 执行高频注意力: (B,H,W,C) --> (B,H,W,h_dim); 此时,C=h_dim,因为所有的注意力头都分配给了高频注意力
            x = self.hifi(x)
            return x.reshape(B, N, C)

        # 执行高频注意力: (B,H,W,C) --> (B,H,W,h_dim)
        hifi_out = self.hifi(x)
        # 执行低频注意力: (B,H,W,C) --> (B,H,W,l_dim)
        lofi_out = self.lofi(x)

        # 在通道方向上拼接高频注意力和低频注意力的输出: (B,H,W,h_dim+l_dim)== (B,H,W,C)
        x = torch.cat((hifi_out, lofi_out), dim=-1)
        # 将输出重塑为与输入相同的shape: (B,H,W,C)-->(B,N,C)
        # x = x.reshape(B, N, C)
        # print(x.shape)
        x = x.reshape(B, C, H, W)

        print(x.shape)

        return x

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

class crossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(crossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x, y):
        b,c,h,w = x.shape
        #print(x.shape, y)

        kv = self.kv_dwconv(self.kv(x))
        k,v = kv.chunk(2, dim=1)
        q = self.q(y)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        #print(attn.shape)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_weight = nn.Parameter(torch.tensor(0.5))  # learnable scalar between 0 and 1
        self.HiLo = HiLoAttention(dim, num_heads, bias)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.skip_scale= nn.Parameter(torch.ones(1,dim,1,1))
        self.skip_scale2= nn.Parameter(torch.ones(1,dim,1,1))

    def forward(self, x):
        # print(self.attn(self.norm1(x)).shape)
        # print((x*self.skip_scale).shape)
        B, C, H, W = x.shape

        attn_out = self.attn(self.norm1(x))
        hilo_out = self.HiLo(self.norm1(x)).reshape(B, C, H, W)
        attn_fused = self.attn_weight * attn_out + (1 - self.attn_weight) * hilo_out
        x = x * self.skip_scale + attn_fused
        x = x * self.skip_scale2 + self.ffn(self.norm2(x))

        return x
    
class crossTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(crossTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = crossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.skip_scale= nn.Parameter(torch.ones(1,dim,1,1))
        self.skip_scale2= nn.Parameter(torch.ones(1,dim,1,1))

    def forward(self, input):
        x = input[0]
        y = input[1]
        x = x*self.skip_scale + self.attn(self.norm1(x), y)
        x = x*self.skip_scale2 + self.ffn(self.norm2(x))

        return [x,y]
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
    
class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

# 一个轻量级的​​光照先验估计网络​​，主要用于从输入图像中提取光照相关特征并进行增强处理
class F_light_prior_estimater(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=16, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(F_light_prior_estimater, self).__init__()

        self.conv1 = nn.Conv2d(16, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        _,_,h,w=img.shape
        depth = min(h,w)//10
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()

        f0 = dct_2d(img,norm='ortho')
        f1 = low_pass(f0, 3*depth).cuda()
        f2 = low_pass_and_shuffle(f0, depth).cuda()
        f3 = high_pass(low_pass(f0, 4*depth), 2*depth).cuda()
        f4 = high_pass(f0, 5*depth).cuda()
        ff = torch.cat([f1,f2,f3,f4], dim=1)
        ff = idct_2d(ff, norm='ortho')

        input = torch.cat([img,mean_c,ff], dim=1)

        x_1 = self.conv1(input)

        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class L_net(nn.Module):
    def __init__(self, num=48, num_heads = 1, num_blocks = 2,inp_channels = 3, out_channels = 1, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(L_net, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(*[TransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Conv2d(num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input):
        out = self.patch_embed(input)
        out = self.encoder(out)
        out = self.output(out)
        return torch.sigmoid(out) + torch.mean(input,dim=1)
        #return torch.sigmoid(out)


class R_net(nn.Module):
    def __init__(self, num=64, num_heads = 1, num_blocks = 2,inp_channels = 3, out_channels = 3, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(R_net, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(*[crossTransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Conv2d(num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    

    def forward(self, input, fea):
        x = self.patch_embed(input)
        x,_ = self.encoder([x,fea])
        #x =self.encoder(x)
        out = self.output(x)
        return torch.sigmoid(out) + input
        #return torch.sigmoid(out)

class N_net(nn.Module):
    def __init__(self, num=64, num_heads = 1, num_blocks = 2,inp_channels = 3, out_channels = 3, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(N_net, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(*[TransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Conv2d(num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    

    def forward(self, input):
        x = self.patch_embed(input)
        x = self.encoder(x)
        out = self.output(x)
        return torch.sigmoid(out) + input
        #return torch.sigmoid(out)

class L_enhance_net(nn.Module):
    def __init__(self, in_channel = 1, num = 32, num_heads = 1, num_blocks = 2, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(L_enhance_net, self).__init__()
        self.patch_embed = OverlapPatchEmbed(in_channel, num)
        self.encoder = nn.Sequential(*[TransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU()
        )
        self.lin = nn.AdaptiveAvgPool2d(1)
        self.tail = nn.Sequential(
            nn.Conv2d(num, num//2, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(num//2,1,kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )

    def forward(self, input):
        out = self.head(input)
        #out = self.patch_embed(input)
        #out = self.encoder(out)
        #print(out)
        out = self.lin(out)
        #print(out)
        out = self.tail(out)
        #print(out)
        return out


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()        
        self.L_net = L_net(num=48)
        self.R_net = R_net(num=64)
        self.N_net = N_net(num=64)
        self.illp = F_light_prior_estimater(n_fea_middle=64)
        #self.D_net = R_net(num=64)
        #self.D_net = D_net(in_ch=3, out_ch=3, base_ch=64, num_module=4)
        self.L_enhance_net = L_enhance_net(in_channel = 1, num = 32)
        #self.L_enhance = L_enhance_net_new(in_channel=1, num=32)

    def forward(self, input):
        x = self.N_net(input)
        y,_ = self.illp(x)
        # Decompose-Net (L:光照图，R:反射图)
        L = self.L_net(x)
        R = self.R_net(x,y)
        #R2 = self.D_net(R1)

        #光照增强的参数
        alpha = self.L_enhance_net(L)
        #el = torch.pow(L,alpha)
        I = torch.pow(L,alpha) * R
        """ if Training:
            noise_adder = AugmentNoise(style='poisson10')
            DI = noise_adder.add_train_noise(I)
            DI = self.D_net(DI)
        else:
            DI = self.D_net(I) """
        return L, L, R, x, I


if __name__=="__main__":
    a = torch.rand(1,3,256,256).cuda()
    model= net().cuda()
    flops, params = profile(model, inputs=(a, ))
    print(flops,params)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        L, el, R, X ,I = model(a)  # 模型推理
    print(L.shape)
    print(el.shape)
    print(R.shape)
    print(X.shape)
    print(I.shape)
    end_time = time.time()

    print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms") 

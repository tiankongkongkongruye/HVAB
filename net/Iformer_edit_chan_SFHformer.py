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

from net.SFHformer import SFHformer_s, Backbone

operation_seed_counter = 0


################################################
# 修改注意力机制需要增加的函数
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, bias=True)
    def forward(self, x):
        x = self.proj(x)
        return x
class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=True),
        )
    def forward(self, x):
        x = self.proj(x)
        return x
class PatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim * patch_size ** 2, kernel_size=1, bias=False),
            nn.PixelShuffle(patch_size),
        )
    def forward(self, x):
        x = self.proj(x)
        return x
class DownSample(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim
        self.proj = nn.Sequential(nn.Conv2d(input_dim, input_dim * 2, kernel_size=2, stride=2))
    def forward(self, x):
        x = self.proj(x)
        return x
class FFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(FFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )
        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
        )
    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)
        return x
class TokenMixer_For_Local(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(TokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim//2
        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.cat([cd1, cd2], dim=1)
        return x
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.fdc = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2 * self.groups,
                                                        kernel_size=1, stride=1, padding=0, groups=self.groups, bias=True)
        self.weight = nn.Sequential(
             nn.Conv2d(in_channels=in_channels * 2, out_channels=self.groups, kernel_size=1, stride=1, padding=0),
             nn.Softmax(dim=1)
        )
        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                                        padding=1, stride=1, groups=in_channels * 2,bias=True)
    def forward(self, x):
        batch, c, h, w = x.size()
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted
        dy_weight = self.weight(ffted)
        ffted = self.fdc(ffted).view(batch, self.groups, 2*c, h, -1)  # (batch, c*2, h, w/2+1)
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        return output
class TokenMixer_For_Gloal(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(TokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)
    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x+x0)
        return x
class Mixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_local=TokenMixer_For_Local,
            token_mixer_for_gloal=TokenMixer_For_Gloal,
    ):
        super(Mixer, self).__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim=self.dim,)
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim,)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*dim, 2*dim//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dim//2, 2*dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, 2*dim, 1),
        )
    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)
        return x





###################################################

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
 
# 修改的注意力
class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=1,
            norm_layer=nn.BatchNorm2d,
            token_mixer=Mixer,
            ffn_expansion_factor = 2.66, 
            bias = False, 
            LayerNorm_type = 'WithBias'
    ):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=self.dim)
        self.ffn = FFN(dim=self.dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy
        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + copy
        return x

##################################################################################
    
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

from torch.nn import init
# 增加的通道注意力
class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B,C)-->fc-->(B,C)-->(B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # scale: (B,C,H,W) * (B, C, 1, 1) == (B,C,H,W)
        out = x * y
        return out


class L_net(nn.Module):
    def __init__(self, num=48, num_heads = 1, num_blocks = 2,inp_channels = 3, out_channels = 1, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(L_net, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(*[TransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.seAttention = SEAttention(channel = num)
        self.output = nn.Conv2d(num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input):
        out = self.patch_embed(input)
        out = self.encoder(out)
        out = self.seAttention(out) # 增加了通道注意力机制
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

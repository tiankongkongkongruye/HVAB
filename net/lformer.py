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
import clip
from torchvision import transforms
import numpy as np
import pywt

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



##########################################################################
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
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.skip_scale= nn.Parameter(torch.ones(1,dim,1,1))
        self.skip_scale2= nn.Parameter(torch.ones(1,dim,1,1))

    def forward(self, x):
        x = x*self.skip_scale + self.attn(self.norm1(x))
        x = x*self.skip_scale2 + self.ffn(self.norm2(x))

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


# 可学习小波包变换
class LearnableWaveletPacket(nn.Module):
    def __init__(self, levels=3):
        super(LearnableWaveletPacket, self).__init__()
        self.levels = levels
        # 可学习的小波滤波器参数
        self.dec_lo = nn.Parameter(torch.Tensor([0.7071, 0.7071]))  # 低通分解滤波器
        self.dec_hi = nn.Parameter(torch.Tensor([0.7071, -0.7071]))  # 高通分解滤波器
        self.rec_lo = nn.Parameter(torch.Tensor([0.7071, 0.7071]))  # 低通重构滤波器
        self.rec_hi = nn.Parameter(torch.Tensor([0.7071, -0.7071]))  # 高通重构滤波器
        
        # 调整大小到224x224用于CLIP
        self.resize = transforms.Resize((224, 224))
        
    def forward(self, x):
        """
        对输入图像进行j级可学习小波包变换
        
        参数:
            x: 输入图像 [B, C, H, W]
            
        返回:
            features: 多尺度特征列表 [F_1, F_2, ..., F_j]
            resized_features: 调整大小后的特征，用于CLIP [F_1_224, F_2_224, ..., F_j_224]
        """
        batch_size, channels, height, width = x.shape
        features = []
        resized_features = []
        
        # 对每个通道单独进行小波变换
        for c in range(channels):
            x_c = x[:, c:c+1, :, :]  # [B, 1, H, W]
            
            # 多级小波包变换
            coeffs = []
            for level in range(self.levels):
                if level == 0:
                    # 第一级变换使用原始图像
                    input_tensor = x_c
                else:
                    # 后续级别使用上一级的低频子带
                    input_tensor = coeffs[-1][0]  # LL子带
                
                # 使用可学习滤波器进行小波变换
                # 水平方向变换
                h_lo = F.conv2d(input_tensor, self.dec_lo.view(1, 1, 1, 2).expand(1, 1, 1, 2), 
                               padding=(0, 1), stride=(1, 2))
                h_hi = F.conv2d(input_tensor, self.dec_hi.view(1, 1, 1, 2).expand(1, 1, 1, 2), 
                               padding=(0, 1), stride=(1, 2))
                
                # 垂直方向变换
                ll = F.conv2d(h_lo, self.dec_lo.view(1, 1, 2, 1).expand(1, 1, 2, 1), 
                             padding=(1, 0), stride=(2, 1))
                lh = F.conv2d(h_lo, self.dec_hi.view(1, 1, 2, 1).expand(1, 1, 2, 1), 
                             padding=(1, 0), stride=(2, 1))
                hl = F.conv2d(h_hi, self.dec_lo.view(1, 1, 2, 1).expand(1, 1, 2, 1), 
                             padding=(1, 0), stride=(2, 1))
                hh = F.conv2d(h_hi, self.dec_hi.view(1, 1, 2, 1).expand(1, 1, 2, 1), 
                             padding=(1, 0), stride=(2, 1))
                
                # 存储当前级别的系数
                coeffs.append((ll, lh, hl, hh))
                
                # 创建当前级别的特征图
                level_feature = torch.cat([ll, lh, hl, hh], dim=1)  # [B, 4, H/2^(level+1), W/2^(level+1)]
                features.append(level_feature)
                
                # 调整大小到224x224用于CLIP
                resized_feature = self.resize(level_feature)  # [B, 4, 224, 224]
                resized_features.append(resized_feature)
        
        return features, resized_features
    
    def inverse_transform(self, coeffs):
        """
        执行逆小波包变换
        
        参数:
            coeffs: 小波系数列表
            
        返回:
            reconstructed: 重建的图像
        """
        # 从最深层开始重建
        for level in range(self.levels-1, -1, -1):
            ll, lh, hl, hh = coeffs[level]
            
            # 垂直方向重建
            h_lo = F.conv_transpose2d(ll, self.rec_lo.view(1, 1, 2, 1).expand(1, 1, 2, 1), 
                                     stride=(2, 1)) + \
                   F.conv_transpose2d(lh, self.rec_hi.view(1, 1, 2, 1).expand(1, 1, 2, 1), 
                                     stride=(2, 1))
            
            h_hi = F.conv_transpose2d(hl, self.rec_lo.view(1, 1, 2, 1).expand(1, 1, 2, 1), 
                                     stride=(2, 1)) + \
                   F.conv_transpose2d(hh, self.rec_hi.view(1, 1, 2, 1).expand(1, 1, 2, 1), 
                                     stride=(2, 1))
            
            # 水平方向重建
            reconstructed = F.conv_transpose2d(h_lo, self.rec_lo.view(1, 1, 1, 2).expand(1, 1, 1, 2), 
                                              stride=(1, 2)) + \
                            F.conv_transpose2d(h_hi, self.rec_hi.view(1, 1, 1, 2).expand(1, 1, 1, 2), 
                                              stride=(1, 2))
            
            # 如果不是最后一级，更新下一级的LL子带
            if level > 0:
                coeffs[level-1] = (reconstructed, coeffs[level-1][1], coeffs[level-1][2], coeffs[level-1][3])
        
        return reconstructed

# CLIP特征提取器
class CLIPFeatureExtractor(nn.Module):
    def __init__(self, clip_model="ViT-B/32"):
        super(CLIPFeatureExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # 冻结CLIP模型参数
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, images):
        """
        提取图像的CLIP特征
        
        参数:
            images: 图像张量 [B, C, 224, 224]
            
        返回:
            features: CLIP特征
        """
        with torch.no_grad():
            features = self.model.encode_image(images)
        return features

# 轻量级跨模态调制器
class CrossModalModulator(nn.Module):
    def __init__(self, feature_dim, embedding_dim):
        super(CrossModalModulator, self).__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # 可学习投影矩阵
        self.Q = nn.Linear(feature_dim, feature_dim)
        self.V = nn.Linear(feature_dim, feature_dim)
        
        # MLP用于生成调制参数
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + embedding_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, F_j, e_j):
        """
        实现轻量级跨模态调制
        
        参数:
            F_j: 多尺度特征 [B, C_j, H, W]
            e_j: CLIP语义嵌入 [B, embedding_dim]
            
        返回:
            F_j_hat: 调制后的特征 [B, C_j, H, W]
        """
        batch_size, channels, height, width = F_j.shape
        
        # 将特征展平为序列
        F_j_flat = F_j.view(batch_size, channels, -1)  # [B, C_j, H*W]
        F_j_flat = F_j_flat.permute(0, 2, 1)  # [B, H*W, C_j]
        
        # 连接特征和嵌入
        e_j_expanded = e_j.unsqueeze(1).expand(-1, F_j_flat.size(1), -1)  # [B, H*W, embedding_dim]
        concat_features = torch.cat([F_j_flat, e_j_expanded], dim=2)  # [B, H*W, C_j+embedding_dim]
        
        # 通过MLP生成调制参数
        gamma_j = self.mlp(concat_features)  # [B, H*W, C_j]
        
        # 应用公式: F_j_hat = F_j + gamma_j ⊙ σ((Q_j*e_j)/√d) * V_j
        Q_j = self.Q(e_j)  # [B, feature_dim]
        V_j = self.V(e_j)  # [B, feature_dim]
        
        # 计算注意力权重
        d = float(self.feature_dim) ** 0.5
        attention_weights = torch.sigmoid(Q_j.unsqueeze(1) / d)  # [B, 1, feature_dim]
        
        # 应用调制
        modulation = gamma_j * attention_weights * V_j.unsqueeze(1)  # [B, H*W, C_j]
        
        # 重塑回原始形状
        modulation = modulation.permute(0, 2, 1).view(batch_size, channels, height, width)  # [B, C_j, H, W]
        F_j_hat = F_j + modulation
        
        return F_j_hat

# 双路径注意力机制
class DualPathAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7):
        super(DualPathAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        # 全局路径参数
        self.global_query = nn.Linear(dim, dim)
        self.global_key = nn.Linear(dim, dim)
        self.global_value = nn.Linear(dim, dim)
        
        # 局部路径参数
        self.local_query = nn.Linear(dim, dim)
        self.local_key = nn.Linear(dim, dim)
        self.local_value = nn.Linear(dim, dim)
        
        # 可学习门控融合
        self.gate_alpha = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # 插值器
        self.interpolate = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        实现双路径注意力机制
        
        参数:
            x: 输入特征 [B, N, C]
            
        返回:
            y: 融合后的特征 [B, N, C]
        """
        B, N, C = x.shape
        
        # 全局路径
        q_global = self.global_query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k_global = self.global_key(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_global = self.global_value(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # 全局注意力
        attn_global = (q_global @ k_global.transpose(-2, -1)) * self.scale
        attn_global = attn_global.softmax(dim=-1)
        y_global = (attn_global @ v_global).transpose(1, 2).reshape(B, N, C)
        
        # 局部路径 - 将特征分割为窗口
        H = W = int(N ** 0.5)  # 假设输入是方形
        x_windows = []
        y_local_windows = []
        
        # 将特征重塑为空间形式
        x_spatial = x.view(B, H, W, C)
        
        # 提取窗口
        for i in range(0, H, self.window_size):
            for j in range(0, W, self.window_size):
                h_end = min(i + self.window_size, H)
                w_end = min(j + self.window_size, W)
                window = x_spatial[:, i:h_end, j:w_end, :].reshape(B, -1, C)
                x_windows.append(window)
        
        # 对每个窗口应用局部注意力
        for window in x_windows:
            Nw = window.shape[1]  # 窗口中的令牌数
            
            q_local = self.local_query(window).reshape(B, Nw, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k_local = self.local_key(window).reshape(B, Nw, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v_local = self.local_value(window).reshape(B, Nw, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            
            attn_local = (q_local @ k_local.transpose(-2, -1)) * self.scale
            attn_local = attn_local.softmax(dim=-1)
            y_local_window = (attn_local @ v_local).transpose(1, 2).reshape(B, Nw, C)
            y_local_windows.append(y_local_window)
        
        # 重建局部特征
        y_local_list = []
        window_idx = 0
        for i in range(0, H, self.window_size):
            for j in range(0, W, self.window_size):
                h_end = min(i + self.window_size, H)
                w_end = min(j + self.window_size, W)
                h_size = h_end - i
                w_size = w_end - j
                
                y_local_list.append((i, j, h_size, w_size, y_local_windows[window_idx]))
                window_idx += 1
        
        # 初始化空的局部特征图
        y_local = torch.zeros(B, H, W, C, device=x.device)
        
        # 填充局部特征图
        for i, j, h_size, w_size, window in y_local_list:
            y_local[:, i:i+h_size, j:j+w_size, :] = window.view(B, h_size, w_size, C)
        
        # 将局部特征展平
        y_local = y_local.view(B, N, C)
        
        # 插值局部特征
        y_local_interpolated = self.interpolate(y_local)
        
        # 计算门控权重
        concat_features = torch.cat([y_global, y_local_interpolated], dim=-1)
        alpha = self.gate_alpha(concat_features)
        
        # 融合全局和局部特征
        y = alpha * y_global + (1 - alpha) * y_local_interpolated
        
        return y

# 新增的可微分层：DEN-Phy模块
class DifferentiableLayer(nn.Module):
    def __init__(self):
        super(DifferentiableLayer, self).__init__()
        # 初始化可学习参数 κ, σs, σc
        self.kappa = nn.Parameter(torch.tensor(1.0))  # κ > 0
        self.sigma_s = nn.Parameter(torch.tensor(0.1))  # σs > 0
        self.sigma_c = nn.Parameter(torch.tensor(0.1))  # σc > 0
        
    def kappa_function(self, x):
        # 实现 κ[x] = (1+κ)x / (1+κx), κ > 0
        return (1 + self.kappa) * x / (1 + self.kappa * x)
    
    # 使用公式N(σs, σc) = sqrt(σs^2(R ∘ L) + σc^2) ⊙ ε, ε ~ N(0,1)实现 
    def noise_estimation(self, R, L):
        
        # 生成标准正态分布噪声
        epsilon = torch.randn_like(R)
        # 计算噪声方差
        variance = self.sigma_s**2 * (R * L) + self.sigma_c**2
        # 计算噪声
        noise = torch.sqrt(variance) * epsilon
        return noise
        
    def forward(self, R, L):
        # 计算噪声
        N = self.noise_estimation(R, L)
        # 计算增强后的图像 Î = κ[(R + sqrt(σs^2(R ∘ L) + σc^2) ⊙ ε) ∘ L]
        enhanced = self.kappa_function((R + N) * L)
        return enhanced, N
    
    def compute_physical_loss(self, I_hat, I, mu=0.1, nu=0.1):
        """
        计算物理损失函数: L_phy = ||Î - I||_2^2 + μ||∇Î||_1 + ν(||∇σs||_1 + ||∇σc||_1)
        
        参数:
            I_hat: 增强后的图像
            I: 原始输入图像
            mu: 梯度正则化超参数
            nu: 噪声参数梯度正则化超参数
        
        返回:
            loss: 物理损失值
        """
        # 计算重建损失
        recon_loss = torch.mean((I_hat - I) ** 2)
        
        # 计算图像梯度
        grad_x = torch.abs(I_hat[:, :, :, :-1] - I_hat[:, :, :, 1:])
        grad_y = torch.abs(I_hat[:, :, :-1, :] - I_hat[:, :, 1:, :])
        grad_norm = torch.mean(grad_x) + torch.mean(grad_y)
        
        # 计算噪声参数梯度 (这里我们使用参数的绝对值作为简化)
        sigma_s_grad = torch.abs(self.sigma_s)
        sigma_c_grad = torch.abs(self.sigma_c)
        
        # 组合损失
        loss = recon_loss + mu * grad_norm + nu * (sigma_s_grad + sigma_c_grad)
        
        return loss



class net(nn.Module):
    def __init__(self, wavelet_levels=3, clip_model="ViT-B/32", feature_dim=512, embedding_dim=512, window_size=7):
        super(net, self).__init__()        
        self.L_net = L_net(num=48)
        self.R_net = R_net(num=64)
        self.N_net = N_net(num=64)
        self.illp = F_light_prior_estimater(n_fea_middle=64)
        self.L_enhance_net = L_enhance_net(in_channel = 1, num = 32)
        # 添加可微分层
        self.diff_layer = DifferentiableLayer()
        
        # 添加可学习小波包变换
        self.wavelet = LearnableWaveletPacket(levels=wavelet_levels)
        
        # 添加CLIP特征提取器
        self.clip_extractor = CLIPFeatureExtractor(clip_model=clip_model)
        
        # 添加跨模态调制器
        self.cross_modal_modulators = nn.ModuleList([
            CrossModalModulator(feature_dim=feature_dim, embedding_dim=embedding_dim)
            for _ in range(wavelet_levels)
        ])
        
        # 计算特征维度
        self.feature_dim = feature_dim
        
        # 添加双路径注意力机制
        self.dual_path_attention = DualPathAttention(dim=feature_dim, window_size=window_size)
        
        # 添加投影层，用于将特征映射回原始空间
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, input):
        x = self.N_net(input)
        y,_ = self.illp(x)
        # Decompose-Net (L:光照图，R:反射图)
        L = self.L_net(x)
        R = self.R_net(x,y)
        
        # 对输入进行小波包变换
        features, resized_features = self.wavelet(input)
        
        # 提取CLIP特征
        clip_embeddings = []
        for resized_feature in resized_features:
            embedding = self.clip_extractor(resized_feature)
            clip_embeddings.append(embedding)
        
        # 应用跨模态调制
        modulated_features = []
        for i, (feature, embedding) in enumerate(zip(features, clip_embeddings)):
            modulated_feature = self.cross_modal_modulators[i](feature, embedding)
            modulated_features.append(modulated_feature)
        
        # 将调制后的特征展平并连接成标记序列
        batch_size = input.shape[0]
        flattened_features = []
        for feature in modulated_features:
            b, c, h, w = feature.shape
            flattened = feature.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
            flattened_features.append(flattened)
        
        # 连接所有特征
        X = torch.cat(flattened_features, dim=1)  # [B, N, C]
        
        # 应用双路径注意力
        Y = self.dual_path_attention(X)
        
        # 投影回原始空间
        Y_projected = self.projection(Y)
        
        # 重塑为原始特征形状
        reconstructed_features = []
        start_idx = 0
        for feature in modulated_features:
            b, c, h, w = feature.shape
            tokens_count = h * w
            feature_tokens = Y_projected[:, start_idx:start_idx+tokens_count, :]
            reconstructed = feature_tokens.permute(0, 2, 1).view(b, c, h, w)
            reconstructed_features.append(reconstructed)
            start_idx += tokens_count
        
        # 使用逆小波包变换重建图像
        # 这里简化处理，直接使用原始的L和R
        alpha = self.L_enhance_net(L)
        el = torch.pow(L, alpha)

        # 使用可微分层生成增强图像
        I, N = self.diff_layer(R, el)
        
        return L, el, R, N, I, modulated_features, Y
        
    def compute_frequency_loss(self, modulated_features, clip_embeddings):
        """
        计算频率损失：CLIP引导的伪目标与调制的多尺度特征二次范式的平方
        """
        loss = 0.0
        for feature, embedding in zip(modulated_features, clip_embeddings):
            # 将特征展平
            b, c, h, w = feature.shape
            flat_feature = feature.view(b, c, -1)
            
            # 计算特征的二次范数
            feature_norm = torch.norm(flat_feature, p=2, dim=2)
            
            # 计算CLIP嵌入的二次范数
            embedding_norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
            
            # 计算损失
            level_loss = torch.mean((feature_norm - embedding_norm) ** 2)
            loss += level_loss
            
        return loss / len(modulated_features)
    
    def compute_consistency_loss(self, global_path, local_path):
        """
        计算全局路径和局部路径之间的一致性损失
        """
        return torch.mean(torch.abs(global_path - local_path))
    
    def compute_reconstruction_loss(self, R, L, I):
        """
        计算重建损失：||R̂ ∘ L^α - I||₁ + ||∇L̂||₁ + ||∇R̂||₂²
        """
        # 计算重建误差
        recon_error = torch.mean(torch.abs(R * L - I))
        
        # 计算L的梯度
        L_grad_x = torch.abs(L[:, :, :, :-1] - L[:, :, :, 1:])
        L_grad_y = torch.abs(L[:, :, :-1, :] - L[:, :, 1:, :])
        L_grad = torch.mean(L_grad_x) + torch.mean(L_grad_y)
        
        # 计算R的梯度
        R_grad_x = torch.abs(R[:, :, :, :-1] - R[:, :, :, 1:])
        R_grad_y = torch.abs(R[:, :, :-1, :] - R[:, :, 1:, :])
        R_grad = torch.mean(R_grad_x ** 2) + torch.mean(R_grad_y ** 2)
        
        return recon_error + L_grad + R_grad


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

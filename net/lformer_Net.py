import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pdb import set_trace as stx
import numbers
from einops import rearrange
from dct_util import dct_2d,idct_2d,low_pass,low_pass_and_shuffle,high_pass
from thop import profile
import time
import clip
from torchvision.transforms import Resize
import math

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


# ========================================================
# 创建多层可学习小波变换层
class LearnableWaveletTransform(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, levels=3):
        super(LearnableWaveletTransform, self).__init__()
        self.levels = levels
        self.transforms = nn.ModuleList()
        
        # 为每一层创建可学习的小波变换
        for j in range(levels):
            # 使用卷积层模拟小波变换
            self.transforms.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))
    
    def forward(self, x):
        # 存储每一层的特征
        features = []
        
        # 对每一层应用变换
        for j in range(self.levels):
            features.append(self.transforms[j](x))
            
        return features

# 特征编码语义嵌入, 轻量级跨模态调制器计算
class CrossModalModulator(nn.Module):
    def __init__(self, feature_dim=16, embedding_dim=512):
        super(CrossModalModulator, self).__init__()
        self.resize = Resize([224, 224])
        
        # 加载预训练的CLIP-VIT模型
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        
        # 冻结CLIP模型参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 为每层j创建独立的可学习投影矩阵Q_j和V_j
        self.num_levels = 3  # 定义层数
        self.Q_layers = nn.ModuleList([nn.Linear(embedding_dim, feature_dim) for _ in range(self.num_levels)])
        self.V_layers = nn.ModuleList([nn.Linear(embedding_dim, feature_dim) for _ in range(self.num_levels)])
        self.T_layers = nn.ModuleList([nn.Linear(embedding_dim, feature_dim) for _ in range(self.num_levels)])
        self.P_layers = nn.ModuleList([nn.Conv2d(feature_dim, 3, kernel_size=1, padding=0) for _ in range(self.num_levels)])
        
        # 使用单个MLP用于生成γ
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # 归一化因子
        self.sqrt_d = math.sqrt(embedding_dim)
        
    def forward(self, features, input_rgb=None):
        modulated_features = []
        clip_targets = []
        for j, F_j in enumerate(features):
            # 生成伪RGB以提取对应层的CLIP嵌入
            pseudo_rgb = self.P_layers[j](F_j)
            resized = self.resize(pseudo_rgb).float()
            with torch.no_grad():
                e_j = self.clip_model.encode_image(resized).float()
            e_j = (e_j - e_j.mean(dim=1, keepdim=True)) / (e_j.std(dim=1, keepdim=True) + 1e-6)
            avg_F_j = F_j.mean(dim=[2, 3]).float()
            concat_features = torch.cat([avg_F_j, e_j], dim=1)
            gamma_j = self.mlp(concat_features)
            Q_e_j = self.Q_layers[j](e_j) / self.sqrt_d
            activated_Q_e_j = torch.sigmoid(Q_e_j)
            V_e_j = self.V_layers[j](e_j)
            batch_size, channels = F_j.shape[0], F_j.shape[1]
            height, width = F_j.shape[2], F_j.shape[3]
            gamma_j = gamma_j.view(batch_size, channels, 1, 1).expand(-1, -1, height, width).float()
            activated_Q_e_j = activated_Q_e_j.view(batch_size, channels, 1, 1).expand(-1, -1, height, width).float()
            V_map = V_e_j.view(batch_size, channels, 1, 1).expand(-1, -1, height, width).float()
            modulation_term = gamma_j * activated_Q_e_j * V_map
            modulated_F_j = F_j.float() + modulation_term
            modulated_features.append(modulated_F_j)
            # 频率目标: 由 e_j 投影后广播到特征图尺寸
            F_clip_vec = self.T_layers[j](e_j)
            F_clip_map = F_clip_vec.view(batch_size, channels, 1, 1).expand(-1, -1, height, width).float()
            clip_targets.append(F_clip_map)
        self.modulated_features = modulated_features
        self.clip_targets = clip_targets
        return modulated_features

class GLCFormer(nn.Module):
    """
    Global-Local Consistency Transformer (GLC-Former)
    用于处理调制后的小波特征
    """
    def __init__(self, dim=64, num_heads=8, window_size=8, num_levels=3, bias=False):
        super(GLCFormer, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_levels = num_levels
        self.scale = dim ** -0.5  # 缩放因子用于注意力计算
        self.global_mode = 'center'
        self.global_tokens_limit = 8192
        
        # 全局路径参数
        self.W_Q = nn.Linear(dim, dim, bias=bias)
        self.W_K = nn.Linear(dim, dim, bias=bias)
        self.W_V = nn.Linear(dim, dim, bias=bias)
        
        # 局部路径参数（每个窗口）
        self.W_Q_local = nn.Linear(dim, dim, bias=bias)
        self.W_K_local = nn.Linear(dim, dim, bias=bias)
        self.W_V_local = nn.Linear(dim, dim, bias=bias)
        
        # 可学习门控 - 接收拼接的全局和局部特征
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),  # 输入维度是2倍的dim，因为拼接了全局和局部特征
            nn.Sigmoid()
        )
        
        # 逆小波变换（简化实现）
        self.inverse_wavelet = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim*4, kernel_size=3, padding=1, bias=bias),
                nn.PixelShuffle(2)  # 上采样
            ) for _ in range(num_levels-1)
        ])
        # 轻量输出解码头
        self.to_rgb = nn.Sequential(
            nn.Conv2d(dim, 32, kernel_size=3, padding=1, bias=bias), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=bias), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, padding=0, bias=bias), nn.Sigmoid()
        )
        
    def _flatten_and_concat_features(self, modulated_features):
        """将多尺度特征展平并拼接成单一标记序列"""
        batch_size = modulated_features[0].shape[0]
        tokens_list = []
        
        for feature in modulated_features:
            B, C, H, W = feature.shape
            # 将特征从 [B, C, H, W] 转换为 [B, H*W, C]
            tokens = feature.flatten(2).transpose(1, 2)
            tokens_list.append(tokens)
        
        # 拼接所有特征 [B, N, C] 其中 N = sum(H_j*W_j)
        X = torch.cat(tokens_list, dim=1)
        return X
    
    def _partition_windows(self, x):
        """将输入序列按固定token数切分为窗口，自动padding以避免形状错误"""
        B, N, C = x.shape
        window_tokens = self.window_size * self.window_size
        num_windows = math.ceil(N / window_tokens)
        pad_tokens = num_windows * window_tokens - N
        if pad_tokens > 0:
            pad = x.new_zeros(B, pad_tokens, C)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x
        windows = x_padded.view(B, num_windows, window_tokens, C)
        return windows, N
    
    def _get_window_centers(self, windows):
        B, num_windows, window_tokens, C = windows.shape
        centers = windows.mean(dim=2)  # [B, num_windows, C]
        return centers
    def _global_attention(self, x):
        """实现全局路径注意力机制，支持 exact/center/pooled 三模式"""
        B, N, C = x.shape
        mode = getattr(self, 'global_mode', 'center')
        limit = getattr(self, 'global_tokens_limit', 8192)
        if mode == 'exact' and N <= limit:
            Q = self.W_Q(x); K = self.W_K(x); V = self.W_V(x)
            attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            return torch.matmul(attn, V) * x
        if mode == 'pooled' or (mode == 'exact' and N > limit):
            pool = max(1, N // limit)
            # 平均池化到 <= limit 的长度
            x_pool = x.view(B, N//pool, pool, C).mean(dim=2)
            Q = self.W_Q(x_pool); K = self.W_K(x_pool); V = self.W_V(x_pool)
            attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            y_pool = torch.matmul(attn, V) * x_pool  # [B, N//pool, C]
            # 重复回原长度
            y_global = y_pool.unsqueeze(2).repeat(1, 1, pool, 1).reshape(B, -1, C)
            y_global = y_global[:, :N, :]
            return y_global
        # center 模式
        windows, original_n = self._partition_windows(x)
        centers = self._get_window_centers(windows)  # [B, num_windows, C]
        Q = self.W_Q(centers); K = self.W_K(centers); V = self.W_V(centers)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        y_centers = torch.matmul(attn, V) * centers  # [B, num_windows, C]
        # 广播到每个窗口 token
        Bc, num_windows, Cc = y_centers.shape
        window_tokens = self.window_size * self.window_size
        y_broadcast = y_centers.unsqueeze(2).repeat(1, 1, window_tokens, 1)
        y_global = y_broadcast.reshape(Bc, num_windows * window_tokens, Cc)
        y_global = y_global[:, :original_n, :]
        return y_global
    
    def _local_attention_per_window(self, x_w):
        """对单个窗口应用局部注意力"""
        B, window_tokens, C = x_w.shape
        
        # 计算Q, K, V
        Q_w = self.W_Q_local(x_w)  # [B, window_tokens, C]
        K_w = self.W_K_local(x_w)  # [B, window_tokens, C]
        V_w = self.W_V_local(x_w)  # [B, window_tokens, C]
        
        # 计算注意力分数: Y_local^w = Softmax(X_wW_Q^w(X_wW_K^w)^T/√C)X_wW_V^w
        attn_w = torch.matmul(Q_w, K_w.transpose(-2, -1)) * self.scale  # [B, window_tokens, window_tokens]
        attn_w = F.softmax(attn_w, dim=-1)
        
        # 应用注意力并乘以x_w，符合公式
        y_local_w = torch.matmul(attn_w, V_w) * x_w  # [B, window_tokens, C]
        
        return y_local_w
    
    def _local_attention(self, windows):
        """实现局部路径注意力机制（所有窗口）"""
        B, num_windows, window_tokens, C = windows.shape
        y_local_windows = []
        
        # 对每个窗口应用局部注意力
        for w in range(num_windows):
            x_w = windows[:, w]  # [B, window_tokens, C]
            y_local_w = self._local_attention_per_window(x_w)
            y_local_windows.append(y_local_w)
        
        # 重新组合所有窗口的输出 [B, num_windows, window_tokens, C]
        y_local_windows = torch.stack(y_local_windows, dim=1)
        
        return y_local_windows
    
    def _interpolate_local_features(self, y_local_windows, target_n):
        """将局部窗口输出拼接为序列，并裁剪到原始token长度"""
        B, num_windows, window_tokens, C = y_local_windows.shape
        y_local = y_local_windows.reshape(B, num_windows * window_tokens, C)
        if y_local.shape[1] > target_n:
            y_local = y_local[:, :target_n, :]
        return y_local
    
    def _feature_fusion(self, y_global, y_local):
        """使用可学习门控制特征融合"""
        # 计算门控值 α = σ(W_α·[Y_global; Interpolate({Y_local^w})])
        # 拼接全局和局部特征
        concat_features = torch.cat([y_global, y_local], dim=-1)  # [B, N, 2*C]
        alpha = self.gate(concat_features)  # [B, N, C]
        
        # 融合特征 Y = α⊙Y_global + (1-α)⊙Interpolate({Y_local^w})
        y = alpha * y_global + (1 - alpha) * y_local
        # 缓存门控以供一致性相关分析
        self.alpha = alpha
        
        return y
    
    def _reshape_to_feature_maps(self, x, shapes):
        """将序列重塑为特征图列表"""
        B, N, C = x.shape
        
        # 计算每个尺度特征的token数量
        token_counts = [shape[2] * shape[3] for shape in shapes]
        
        # 分割序列
        start_idx = 0
        feature_maps = []
        
        for i, count in enumerate(token_counts):
            # 提取当前尺度的tokens
            tokens = x[:, start_idx:start_idx+count, :]  # [B, H_i*W_i, C]
            
            # 重塑为特征图
            H, W = shapes[i][2], shapes[i][3]
            feature_map = tokens.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
            feature_maps.append(feature_map)
            
            start_idx += count
        
        return feature_maps
    
    def _inverse_wavelet_transform(self, y, original_shapes):
        """逆小波包变换重建"""
        # 将序列y重塑为特征图列表
        B, N, C = y.shape
        
        # 计算每个尺度特征的token数量
        token_counts = [shape[2] * shape[3] for shape in original_shapes]
        
        # 分割序列
        start_idx = 0
        feature_maps = []
        
        for i, count in enumerate(token_counts):
            # 提取当前尺度的tokens
            tokens = y[:, start_idx:start_idx+count, :]  # [B, H_i*W_i, C]
            
            # 重塑为特征图
            H, W = original_shapes[i][2], original_shapes[i][3]
            feature_map = tokens.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
            feature_maps.append(feature_map)
            
            start_idx += count
        
        # 开始逆小波变换重建
        outputs = [feature_maps[0]]  # 最低分辨率特征
        
        # 从低分辨率到高分辨率逐步重建
        x = feature_maps[0]
        for i in range(len(self.inverse_wavelet)):
            x = self.inverse_wavelet[i](x)  # 上采样
            
            # 如果有对应的高分辨率特征，则添加
            if i+1 < len(feature_maps):
                # 确保尺寸匹配
                if x.shape[2:] != feature_maps[i+1].shape[2:]:
                    x = F.interpolate(x, size=feature_maps[i+1].shape[2:], 
                                     mode='bilinear', align_corners=False)
                x = x + feature_maps[i+1]  # 残差连接
            
            outputs.append(x)
        
        return outputs
        
    def forward(self, modulated_features):
        # 保存原始特征形状
        original_shapes = [f.shape for f in modulated_features]
        
        # 1. 将多尺度特征展平并拼接成单一标记序列
        X = self._flatten_and_concat_features(modulated_features)  # [B, N, C]
        
        # 2. 将输入X分割为K×K个互不重叠的窗口
        windows, original_n = self._partition_windows(X)
        
        # 3. 应用全局路径注意力
        y_global = self._global_attention(X)  # [B, N, C]
        
        # 4. 应用局部路径注意力（每个窗口）
        y_local_windows = self._local_attention(windows)  # [B, K*K, window_size*window_size, C]
        
        # 将局部特征插值回原始形状
        y_local = self._interpolate_local_features(y_local_windows, original_n)  # [B, N, C]
        
        # 5. 使用可学习门控制特征融合
        y = self._feature_fusion(y_global, y_local)  # [B, N, C]
        # 缓存以供一致性损失使用
        self.y_global = y_global
        self.y_local = y_local
        
        # 6. 直接使用融合特征进行逆小波包变换重建
        output_features = self._inverse_wavelet_transform(y, original_shapes)
        
        return output_features

# ========================================================

class RetinexModule(nn.Module):
    """基于Retinex理论的并行模块"""
    def __init__(self, channels=3):
        super(RetinexModule, self).__init__()
        
        # 可学习参数：σs和σc控制SDN强度
        self.sigma_s = nn.Parameter(torch.tensor(0.1))  # 初始化为较小值
        self.sigma_c = nn.Parameter(torch.tensor(0.1))  # 初始化为较小值
        
        # 可学习参数：κ控制CRF函数
        self.kappa = nn.Parameter(torch.tensor(0.5))  # 初始化为中等值
        
        # 用于估计光照分量L的卷积网络
        self.illumination_net = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )
        
    def noise_model(self, R, L):
        """实现噪声模型 N(σs, σc) = sqrt(σs^2 * (R ○ L) + σc^2 ○ ε), ε ~ N(0, 1)"""
        batch_size, channels, height, width = R.shape
        
        # 生成标准正态分布噪声
        epsilon = torch.randn(batch_size, channels, height, width, device=R.device)
        
        # 计算噪声
        # R ○ L 表示R和L的元素级乘积
        illumination_dependent_noise = self.sigma_s ** 2 * (R * L)
        # σc^2 ○ ε 表示σc^2与ε的元素级乘积
        constant_noise = self.sigma_c ** 2 
        
        # 组合噪声
        noise = torch.sqrt(illumination_dependent_noise + constant_noise)* epsilon
        
        return noise
    
    def crf_function(self, x):
        """实现CRF函数 κ[x] = (1 + κ)x / (1 + κx), κ > 0"""
        # 确保kappa为正值
        kappa_pos = torch.abs(self.kappa) + 1e-6
        
        # 应用CRF函数
        return (1 + kappa_pos) * x / (1 + kappa_pos * x)
    def inverse_crf(self, y):
        """近似逆CRF: 给定 y=κ[x]，求 x"""
        kappa_pos = torch.abs(self.kappa) + 1e-6
        denom = (1 + kappa_pos) - kappa_pos * y
        x = y / (denom + 1e-6)
        return torch.clamp(x, 0, 1)
    
    def reconstruct_image(self, R, L, noise):
        """实现图像重建 Î = κ[(R + N(σs, σc)) ○ L]"""
        # 计算重建图像
        # (R + noise) ○ L 表示(R + noise)与L的元素级乘积
        enhanced = (R + noise) * L
        # 应用CRF函数
        reconstructed = self.crf_function(enhanced)
        
        return reconstructed
    
    def forward(self, input_image):
        """前向传播
        Args:
            input_image: 输入的低光图像I
        Returns:
            reconstructed_image: 重建的增强图像Î
            R: 反射层
            L: 光照层
            noise: 噪声模型输出
        """
        # 估计光照分量L
        L = self.illumination_net(input_image)
        
        # 根据Retinex理论，I = R * L，因此R = I / L
        # 为避免除零错误，添加一个小的epsilon
        epsilon = 1e-6
        R = input_image / (L + epsilon)
        
        # 裁剪R到合理范围
        R = torch.clamp(R, 0, 1)
        
        # 计算噪声
        noise = self.noise_model(R, L)
        
        # 重建图像
        reconstructed_image = self.reconstruct_image(R, L, noise)
        
        return reconstructed_image, R, L, noise

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()        
        # self.L_net = L_net(num=48)
        # self.R_net = R_net(num=64)
        # self.N_net = N_net(num=64)
        # self.illp = F_light_prior_estimater(n_fea_middle=64)
        # self.L_enhance_net = L_enhance_net(in_channel = 1, num = 32)
        
        # 添加可学习小波包变换和跨模态调制器
        self.wavelet_transform = LearnableWaveletTransform(in_channels=3, out_channels=16, levels=3)
        self.cross_modal_modulator = CrossModalModulator(feature_dim=16, embedding_dim=512)
        self.glc_former = GLCFormer(dim=16, num_heads=8, window_size=8, num_levels=3, bias=False)
        
        # 添加基于Retinex理论的并行模块
        self.retinex_module = RetinexModule(channels=3)
        # 学习式门控融合
        self.gating_net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, padding=0), nn.Sigmoid()
        )

    def forward(self, input):
        # 原始处理流程
        # x = self.N_net(input)
        # y,_ = self.illp(x)
        # L = self.L_net(x)
        # R = self.R_net(x,y)
        # alpha = self.L_enhance_net(L)
        # I = torch.pow(L,alpha) * R
        
        # 应用可学习小波包变换
        wavelet_features = self.wavelet_transform(input)
        modulated_features = self.cross_modal_modulator(wavelet_features, input_rgb=input)
        fused_features = self.glc_former(modulated_features)
        top_feat = fused_features[-1]
        feat_to_img = self.glc_former.to_rgb(top_feat)
        retinex_output, R, L, noise = self.retinex_module(input)
        mode = getattr(self, 'mode', 'hybrid')
        if mode == 'den_phy':
            I = retinex_output
        elif mode == 'cmfsc':
            I = feat_to_img
        elif mode == 'full':
            I = self.retinex_module.inverse_crf(feat_to_img)
        else:
            gate = self.gating_net(torch.cat([L, feat_to_img], dim=1))
            I = gate * retinex_output + (1 - gate) * feat_to_img
        el = L
        X = input
        return L, el, R, X, I

""" a = torch.rand(1,3,128,128).cuda()
model= net().cuda()
flops, params = profile(model, inputs=(a, ))
print(flops,params)
model.eval()
start_time = time.time()
with torch.no_grad():
    _,_,_,_ = model(a)  # 模型推理
end_time = time.time()

print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms") """

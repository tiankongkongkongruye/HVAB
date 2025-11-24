import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
from torchvision import models


operation_seed_counter = 0
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = F.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)

def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

# 增加噪声扰动，增强模型的泛化能力
def pair_downsampler(img):
    #img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c,1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c,1, 1, 1)

    # 最大池化（2x2窗口，步长2）
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # 增加轻微噪声
    # noise = torch.randn_like(img) * 0.02
    # blur = F.avg_pool2d(img, 3, stride=1, padding=1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    # output3 = max_pool(img)

    # return output1, output2, output3
    return output1, output2

def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    return gradient_h, gradient_w


# 新加的一些损失
class ColorConstancyLoss(nn.Module):
    """Color Constancy Loss"""

    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        drg = torch.pow(mr - mg, 2)
        drb = torch.pow(mr - mb, 2)
        dgb = torch.pow(mb - mg, 2)
        k = torch.pow(
            torch.pow(drg, 2) + torch.pow(drb, 2) + torch.pow(dgb, 2), 0.5)
        return k


class ExposureLoss(nn.Module):
    """Exposure Loss"""

    def __init__(self, patch_size, mean_val):
        super(ExposureLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        return torch.mean(torch.pow(
            mean - torch.FloatTensor([self.mean_val]).cuda(), 2
        ))


class IlluminationSmoothnessLoss(nn.Module):
    """Illumination Smoothing Loss"""

    def __init__(self, loss_weight=1):
        super(IlluminationSmoothnessLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class SpatialConsistancyLoss(nn.Module):
    """Spatial Consistancy Loss"""

    def __init__(self):
        super(SpatialConsistancyLoss, self).__init__()

        kernel_left = torch.FloatTensor(
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor(
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor(
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor(
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        d_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        d_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        d_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        d_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        d_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        d_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        d_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        d_left = torch.pow(d_org_left - d_enhance_left, 2)
        d_right = torch.pow(d_org_right - d_enhance_right, 2)
        d_up = torch.pow(d_org_up - d_enhance_up, 2)
        d_down = torch.pow(d_org_down - d_enhance_down, 2)
        return d_left + d_right + d_up + d_down



def tv_loss(illumination):
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    loss_h = gradient_illu_h
    loss_w = gradient_illu_w
    loss = loss_h.mean() + loss_w.mean()
    return loss

def C_loss(R1, R2):
    loss = torch.nn.MSELoss()(R1, R2) 
    return loss

def R_loss(L1, R1, im1, X1):
    max_rgb1, _ = torch.max(im1, 1)
    max_rgb1 = max_rgb1.unsqueeze(1) 
    loss1 = torch.nn.MSELoss()(L1*R1, X1) + torch.nn.MSELoss()(R1, X1/L1.detach())
    loss2 = torch.nn.MSELoss()(L1, max_rgb1) + tv_loss(L1)
    return loss1 + loss2

def P_loss(im1, X1):
    loss = torch.nn.MSELoss()(im1, X1)
    return loss



### NEW ###
# = a============================================================================
# 新增损失函数 1: 互信息损失 (L_MI) 的一个实现
# 使用HSIC (Hilbert-Schmidt Independence Criterion)作为互信息的代理
# ==============================================================================
def rbf_kernel(x, y, sigma=1.0):
    """计算高斯核矩阵"""
    beta = 1. / (2. * sigma)
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-beta * dist_sq)

class MutualInformationLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(MutualInformationLoss, self).__init__()
        self.sigma = sigma

    def forward(self, R, I):
        """
        计算反射图R和光照图I之间的HSIC，以近似互信息
        输入R和I的形状: (b, c, h, w)
        """
        b, c, h, w = R.shape
        
        # 将图像展平为 (b, h*w*c)
        R_flat = R.view(b, -1)
        I_flat = I.view(b, -1)
        
        # 中心化核矩阵
        K_r = rbf_kernel(R_flat, R_flat, self.sigma)
        K_i = rbf_kernel(I_flat, I_flat, self.sigma)
        
        H = torch.eye(b, device=R.device) - 1.0 / b
        
        # 计算HSIC
        hsic = torch.trace(K_r @ H @ K_i @ H) / ((b - 1) ** 2)
        
        return hsic

### NEW ###
# ==============================================================================
# 新增损失函数 2: 变分推断中的KL散度损失
# ==============================================================================
class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, R, I):
        """
        一个简化的KL散度损失实现
        假设先验是均匀分布，促使R和I的像素值分布平滑
        """
        # 对数-softmax可以看作是对数概率分布
        log_R = F.log_softmax(R.view(R.size(0), -1), dim=1)
        log_I = F.log_softmax(I.view(I.size(0), -1), dim=1)
        
        # 假设先验是一个均匀分布，其对数概率是常数
        # 我们促使R和I内部的分布尽可能独立，即KL(R||I)和KL(I||R)都应该小
        # 这里简化为让它们各自远离尖锐的分布
        
        # 计算分布与均匀分布的KL散度
        # KL(q || p_uniform) = E[log q] - E[log p_uniform]
        # 最小化它等价于最大化熵，即让分布更平滑
        
        kl_R = -log_R.mean()
        kl_I = -log_I.mean()

        return kl_R + kl_I

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
def joint_RGB_horizontal(im1, im2):
    if im1.size==im2.size:
        w, h = im1.size
        result = Image.new('RGB',(w*2, h))
        result.paste(im1, box=(0,0))
        result.paste(im2, box=(w,0))      
    return result

def joint_L_horizontal(im1, im2):
    if im1.size==im2.size:
        w, h = im1.size
        result = Image.new('L',(w*2, h))
        result.paste(im1, box=(0,0))
        result.paste(im2, box=(w,0))   
    return result

class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
        
class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d

class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E

# 提高泛化能力：区域和过曝与欠曝
def gamma_correction(images):
    
    gamma = random.uniform(1.3, 1.5)
    # if random.random() < 0.5:
    #     gamma = random.uniform(1.3, 1.5)  # 模拟欠曝图
    # else:
    #     gamma = random.uniform(0.6, 0.9)  # 模拟过曝图
    # gamma = random.uniform(1.2,1.8)
    # gamma = 1.45
    #print(gamma)
    # Apply gamma correction
    corrected_images = images.pow(1 / gamma)
    
    # 不同区域增强与减弱光照
    # luminance = images.mean(dim=1, keepdim=True)  # B×1×H×W
    # dark_mask = (luminance < 0.4).float()
    # gamma_map = 1.0 + 0.6 * dark_mask              # 暗区 gamma ↑
    # corrected_images = images ** (1.0 / gamma_map)

    # 通道不对称，不同rgb来增强
    # gammas = [random.uniform(1.2,1.6), random.uniform(1.0,1.5), random.uniform(1.3,1.7)]
    # corrected_images = torch.stack([
    #     images[:, 0] ** (1.0 / gammas[0]),
    #     images[:, 1] ** (1.0 / gammas[1]),
    #     images[:, 2] ** (1.0 / gammas[2]),
    # ], dim=1)

    # Normalize the pixel values between 0 and 1
    corrected_images = torch.clamp(corrected_images, 0, 1)
    
    return corrected_images

import torchvision.transforms.functional as TF
import torchvision.transforms as T

# 修改的图像增强：非线性 + 模糊 + 噪声 + 色彩抖动
def image_augmentations(images):
    # Gamma Correction
    gamma = random.uniform(1.3, 1.5)
    images = images.pow(1 / gamma)
    
    # Color Jitter（只对RGB图像）
    if random.random() < 0.8:
        # 对于 batch 中每张图分别处理
        for i in range(images.shape[0]):
            img = images[i]
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
            images[i] = img

    # Add Gaussian Blur
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5])
        images = T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))(images)

    # Add Gaussian Noise
    if random.random() < 0.5:
        noise = torch.randn_like(images) * 0.02
        images = images + noise
        images = torch.clamp(images, 0, 1)

    return images


# ====================== 保真度损失 (Fidelity Loss) ======================
# class FidelityLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, z_list, q_list):
#         """
#         z_list: 各递归步骤的输入列表 [z_1, z_2, ..., z_R] (每个元素shape: [B,C,H,W])
#         q_list: 各递归步骤的预测商数列表 [q_1, q_2, ..., q_R]
#         """
#         loss = 0.0
#         for z_r, q_r in zip(z_list, q_list):
#             loss += F.mse_loss(z_r, q_r)  # L2损失
#         return loss / len(z_list)  # 平均所有递归步骤

# ====================== 平滑性损失 (Smoothness Loss) ======================
# class SmoothnessLoss(nn.Module):
#     def __init__(self, window_size=5, sigma=0.1):
#         super().__init__()
#         self.window_size = window_size
#         self.sigma = sigma
#         self.pad = window_size // 2
#         self.unfold = nn.Unfold(kernel_size=window_size, padding=self.pad)

#     def rgb_to_ycbcr(self, image):
#         # RGB转YCbCr（假设输入为[0,1]范围）
#         matrix = torch.tensor([[0.299, 0.587, 0.114],
#                                [-0.1687, -0.3313, 0.5],
#                                [0.5, -0.4187, -0.0813]], device=image.device)
#         ycbcr = torch.einsum('b c h w, c d -> b d h w', image, matrix)
#         ycbcr[:, 1:] += 0.5  # Cb和Cr通道偏移
#         return ycbcr

#     def compute_weights(self, z_r):
#         """ 计算高斯权重矩阵 """
#         # 转换到YCbCr空间
#         z_ycbcr = self.rgb_to_ycbcr(z_r)  # [B,3,H,W]
#         # 提取亮度通道(Y)
#         z_y = z_ycbcr[:, 0:1, :, :]  # [B,1,H,W]
#         # 展开为邻域块
#         patches = self.unfold(z_y)  # [B, window_size^2, H*W]
#         patches = patches.view(z_y.shape[0], self.window_size**2, *z_y.shape[2:])  # [B,25,H,W]
#         # 计算中心像素与邻域的差异
#         center = patches[:, self.window_size**2 // 2, :, :].unsqueeze(1)  # [B,1,H,W]
#         diffs = (patches - center)**2  # [B,25,H,W]
#         # 高斯权重
#         weights = torch.exp(-diffs.sum(dim=1) / (2 * self.sigma**2))  # [B,H,W]
#         return weights

#     def forward(self, q_list, z_list):
#         """
#         q_list: 各递归步骤的商数列表 [q_1, q_2, ..., q_R]
#         z_list: 各递归步骤的输入列表 [z_1, z_2, ..., z_R]
#         """
#         total_loss = 0.0
#         for q_r, z_r in zip(q_list, z_list):
#             B, C, H, W = q_r.shape
#             # 计算权重矩阵
#             weights = self.compute_weights(z_r)  # [B,H,W]
#             # print(weights.shape)
#             # print(q_r.shape)
#             # 计算水平和垂直梯度
#             diff_h = torch.abs(q_r[:, :, 1:, :] - q_r[:, :, :-1, :])  # [B,C,H-1,W]
#             diff_w = torch.abs(q_r[:, :, :, 1:] - q_r[:, :, :, :-1])  # [B,C,H,W-1]
#             # 扩展权重到梯度形状
#             weights_h = weights[:, :-1, :].unsqueeze(1).expand(-1,C,-1,-1)  # [B,C,H-1,W]
#             weights_w = weights[:, :, :-1].unsqueeze(1).expand(-1,C,-1,-1)  # [B,C,H,W-1]

#             # 加权L1损失
#             loss = (weights_h * diff_h).mean() + (weights_w * diff_w).mean()
#             total_loss += loss
#         return total_loss / len(q_list)  # 平均所有递归步骤

# # ====================== 颜色损失 (Color Loss) ======================
# class ColorLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, enhanced_list):
#         """
#         enhanced_list: 各递归步骤的增强图像列表 [x_1, x_2, ..., x_R]
#         """
#         loss = 0.0
#         for x_r in enhanced_list:
#             # 计算各通道均值
#             mean_r = x_r.mean(dim=[2,3])  # [B,3]
#             # 计算所有通道对的差异平方和
#             pairs = [(0,1), (0,2), (1,2)]
#             for (p, q) in pairs:
#                 loss += torch.mean((mean_r[:, p] - mean_r[:, q])**2)
#         return loss / len(enhanced_list)

# # ====================== 感知损失 (Perceptual Loss) ======================


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1,3,1,1)
        self.std = torch.tensor(std).view(1,3,1,1)
    
    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
class PerceptualLoss(nn.Module):
    def __init__(self, layer_name='relu4_3'):
        super().__init__()
        # vgg = models.vgg16(pretrained=True).features
        # layer_index = {
        #     'relu1_2': 4, 'relu2_2': 9, 
        #     'relu3_3': 16, 'relu4_3': 23
        # }[layer_name]
        # self.vgg = nn.Sequential(*list(vgg.children())[:layer_index+1]).to('cuda')
        # self.vgg.eval()
        # 使用ResNet-18浅层特征
        resnet = models.resnet18(pretrained=True).to("cuda")
        self.feature_extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, 
            resnet.maxpool, resnet.layer1
        ).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        

    def forward(self, z_list, q_list):
        """
        z_list: 各递归步骤的输入列表
        q_list: 各递归步骤的商数列表
        """
        loss = 0.0
        for z_r, q_r in zip(z_list, q_list):
            with torch.no_grad():
                # 归一化输入到VGG范围
                z_norm = self.normalize(z_r.detach())
                q_norm = self.normalize(q_r.detach())
                # 提取特征
                feat_z = self.feature_extractor(z_norm)
                feat_q = self.feature_extractor(q_norm)
            # 计算L2距离
            loss += F.mse_loss(feat_z, feat_q)
        return loss / len(z_list)

import os, pyiqa
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torchvision.transforms import Resize
import argparse
import random
import shutil
import clip
import lpips
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
# from net.lformer import net
from net.Iformer_org import net
# from net.Iformer_edit import net
# from net.Iformer_edit_attention_mamba import net
# from net.Iformer_edit_attention_Hist import net
# from net.Iformer_edit_channelattention import net
# from net.Iformer_edit_channelattention_CBAM import net
# from net.Iformer_edit_skNets import net
# from net.Iformer_edit_attention_HiLo import net
# from net.Iformer_edit_attention_MSRBlock import net
# from net.Iformer_edit_chanSp import net
# from net.Iformer_edit_chan_HiLO import net
# from net.Iformer_edit_chan_SFHformer import net
# from net.Iformer_edit_SFHformer import net
# from net.Iformer_edit_crossAttention3 import net
# from net.Iformer_edit_CBAMattention import net
# from net.Iformer_edit_triAttention import net
from data import get_training_set, get_eval_set
from utils import *
from torch.utils.tensorboard import SummaryWriter

import piq

#python main.py --save_folder weights/LOLv1/ --logroot logs/your_log/ --data_train your_train_data --data_val your_val_data --referance_val the_high_quality_val_data --lr 1e-5 --light_patch 64 --loss_weights [1, 0.1, 0.1, 0.5]
#python main.py --save_folder weights/LOLv2/ --logroot logs/your_log/ --data_train your_train_data --data_val your_val_data --referance_val the_high_quality_val_data --lr 5e-6 --light_patch 64 --loss_weights [1, 0.1, 0.1, 0.5]
#python main.py --save_folder weights/SICE/ --logroot logs/your_log/ ---data_train your_train_data --data_val your_val_data --referance_val the_high_quality_val_data --lr 1e-5 --light_patch 32 --loss_weights [1, 0.1, 0.1, 0.01]

#python main_unsup.py --save_folder weights/HTIW/ --logroot logs/HTIW/ --data_train data/HTIW/Train/Low/ --data_val data/HTIW/Test/Low/ --referance_val data/HTIW/Test/Low/ --lr 5e-6 --light_patch 64 --loss_weights [1, 0.1, 0.1, 0.5]
# python main_unsup.py --save_folder weights/middle_log_2/ --logroot logs/middle_log_2/ --data_train data/middle/train/ --data_val data/middle/test --referance_val data/middle/test/ --lr 2e-6 --light_patch 64


# Training settings
parser = argparse.ArgumentParser(description='DE-Net')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=1e-5')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='300', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--gama', type=float, default=1)
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_train', type=str, default='')
parser.add_argument('--data_val', type=str, default='')
parser.add_argument('--referance_val', type=str, default='')
parser.add_argument('--loss_weights', type=list, default=[1, 0.1, 0.1, 0.5])
parser.add_argument('--light_patch', type=int, default=64, help='the patch size of enhancement loss')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--save_folder', default='weights/full_loss/', help='Location to save checkpoint models')
parser.add_argument('--logroot', default='logs/tiaocan_loss0.1', help='Location to save logs')
opt = parser.parse_args()

def seed_torch(seed=opt.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_torch()
cudnn.benchmark = True

# 连续平滑损失
def total_variation_loss(image):
    diff_h = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    diff_w = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    return diff_h.mean() + diff_w.mean()


# 1. 无监督去噪：盲点 MSE（Noise2Void 风格）
class Noise2VoidLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred: torch.Tensor, inp: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        pred: 网络输出 (B,C,H,W)
        inp: 原始带噪图 (B,C,H,W)
        mask: 盲点位置掩码 (B,1,H,W)，True 表示需要计算损失的像素
        """

        # 如果 mask 是 (B,1,H,W)，先去掉 channel=1，再 expand
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask.squeeze(1)  # (B,H,W)
        # 现在 mask: (B,H,W)，扩展到 (B,C,H,W)
        mask = mask.unsqueeze(1).expand_as(pred)  # (B,C,H,W)

        # 仅在 mask 标记的像素位置计算 MSE
        return F.mse_loss(pred[mask], inp[mask])
        # 参考 Noise2Void 实现:contentReference[oaicite:5]{index=5}

# 2. 边缘一致性：GMSD 损失（Gradient Magnitude Similarity Deviation）
class GMSDLoss(nn.Module):
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        # PIQ 中已有实现，可直接调用 
        self.gmsd = piq.GMSDLoss(data_range=data_range, reduction='mean')
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: 归一化到 [0,1] 的图像张量 (B,C,H,W)
        """
        return self.gmsd(pred, target)
        # 参考 PIQ 文档与示例:contentReference[oaicite:6]{index=6}


# 3. 照度均匀性：补丁方差
class UniformityLoss(nn.Module):
    def __init__(self, patch_size: int = 32):
        """
        patch_size: 将图像分割为 patch_size×patch_size 大小的小块
        """
        super().__init__()
        self.ps = patch_size

    def forward(self, illum: torch.Tensor) -> torch.Tensor:
        """
        illum: 单通道或三通道光照图 (B,C,H,W)
        """
        B, C, H, W = illum.shape
        # 分块：unfold 提取补丁:contentReference[oaicite:8]{index=8}
        patches = illum.unfold(2, self.ps, self.ps).unfold(3, self.ps, self.ps)
        # 形状 (B,C, nH, nW, ps, ps)，先对最后两个维度求均值
        patch_means = patches.mean(dim=[4,5])  # (B,C, nH, nW)
        # 对所有补丁的均值计算方差，通道与空间维度一起
        var_p = patch_means.view(B, -1).var(dim=1)  # (B,)
        # 平均各样本损失
        return var_p.mean()
        # 通常希望方差越小，光照越均匀:contentReference[oaicite:9]{index=9}


def generate_blindspot_mask(input_tensor: torch.Tensor,
                            mask_ratio: float = 0.065) -> torch.BoolTensor:
    """
    生成 Noise2Void 风格的盲点掩码。
    
    Args:
        input_tensor (Tensor): 网络输入图像，shape = (B, C, H, W)。
        mask_ratio (float): 盲点像素占总像素的比例，通常 0.05~0.1 之间效果较好:contentReference[oaicite:0]{index=0}。
    
    Returns:
        mask (BoolTensor): 布尔掩码，shape = (B, 1, H, W)，
                           盲点位置为 True，其它位置为 False。
    """
    B, C, H, W = input_tensor.shape
    num_pixels = H * W
    num_mask = int(num_pixels * mask_ratio)
    
    # 创建全 False 的掩码
    mask = torch.zeros((B, 1, H, W), dtype=torch.bool, device=input_tensor.device)
    
    for b in range(B):
        # 在扁平化像素索引中随机选取盲点位置
        idx = torch.randperm(num_pixels, device=input_tensor.device)[:num_mask]
        # 构造每张图的扁平化掩码，然后 reshape 回 (H, W)
        flat_mask = torch.zeros(num_pixels, dtype=torch.bool, device=input_tensor.device)
        flat_mask[idx] = True
        mask[b, 0] = flat_mask.view(H, W)
    
    return mask

def similarity_loss(output, R):
    return F.l1_loss(output, R)

def local_smooth_loss(output):
    Y = 0.299 * output[:, 0:1] + 0.587 * output[:, 1:2] + 0.114 * output[:, 2:3]
    avg = F.avg_pool2d(Y, kernel_size=5, stride=1, padding=2)
    return F.l1_loss(Y, avg)

def global_variance_loss(output):
    Y = 0.299 * output[:, 0:1] + 0.587 * output[:, 1:2] + 0.114 * output[:, 2:3]
    mean = torch.mean(Y, dim=[2, 3], keepdim=True)
    return torch.mean((Y - mean) ** 2)

def train():
    model.train()

    # 示例：组合到训练循环
    denoise_loss = Noise2VoidLoss()
    # edge_loss    = GMSDLoss()
    # unif_loss    = UniformityLoss(patch_size=32)
    loss_print = 0

    l_e = L_exp(opt.light_patch,0.5)
    l_color = L_color()
    l_spa = L_spa()
    l_tv = L_TV()

    # color_loss = ColorConstancyLoss().cuda()
    # spatial_consistency_loss = SpatialConsistancyLoss().cuda()
    exposure_loss = ExposureLoss(patch_size=16, mean_val=0.6).cuda()
    # illumination_smoothing_loss = IlluminationSmoothnessLoss().cuda()

    clipiqa = pyiqa.create_metric("clipiqa", device="cuda")
    brsique = pyiqa.create_metric('brisque', as_loss=False).cuda()


    for iteration, batch in enumerate(training_data_loader, 1):

        input, file1 = batch[0], batch[1]
        input = input.cuda()
        # print(input.shape) [1, 3, 256, 256]
        # 生成两个降采样图像，构建多尺度训练样本
        im1, im2 = pair_downsampler(input)
        # print(im1.shape) [1, 3, 128, 128]
        # print(im2.shape) [1, 3, 128, 128]
        # 非线性增强，模拟不同光照条件
        im2 = gamma_correction(im2)
        # im2 = image_augmentations(im2)
        #im2 = im1.pow(1 / opt.gama)
        #print(im1)
        im1 = im1.cuda()
        im2 = im2.cuda()
        # 光照分量L，反射分量R，去噪中间结果X，增强后的光照图
        L1, el1, R1, X1, I1 = model(im1)
        L2, el2, R2, X2, _ = model(im2)
        _,_,R3,X3,I3 = model(input)
        sub_r1, sub_r2 = pair_downsampler(I3)

        I1 = torch.clamp(I1, 0, 1)
        # el1 = torch.clamp(el1, 0, 1)
        X1 = torch.clamp(X1, 0, 1)
        R1 = torch.clamp(R1, 0, 1)
        R2 = torch.clamp(R2, 0, 1)
        #DI1 = torch.clamp(DI1, 0, 1)
        I3 = torch.clamp(I3, 0, 1)
        X3 = torch.clamp(X3, 0, 1)
        R3 = torch.clamp(R3, 0, 1)
        im3 = torch.clamp(input, 0, 1)

        # output1 = torch.clamp(output1, 0, 1)

        
        
        # 反射一致性约束
        loss1 = C_loss(R1, R2) + C_loss(R1-R2, sub_r1-sub_r2) * 0.5
        #loss1 = C_loss(R1, R2)
        # 物理驱动重建损失
        loss2 = R_loss(L1, R1, im1, X1)
        # 感知损失
        loss3 = P_loss(im1, X1)
        # 复合正则：光照平滑，空间一致性，全变差约束，颜色保真
        # loss5 = opt.loss_weights[0]*l_e(I3) + opt.loss_weights[1]*torch.mean(l_spa(X3, I3)) + opt.loss_weights[2]*l_tv(I3) + opt.loss_weights[3]*torch.mean(l_color(I3))
        loss4 = opt.loss_weights[0]*l_e(I1) + opt.loss_weights[1]*torch.mean(l_spa(X1, I1)) + opt.loss_weights[2]*l_tv(I1) + opt.loss_weights[3]*torch.mean(l_color(I1))
        # loss4 = opt.loss_weights[0]*l_e(R1) + opt.loss_weights[1]*torch.mean(l_spa(X1, R1)) + opt.loss_weights[2]*l_tv(R1) + opt.loss_weights[3]*torch.mean(l_color(R1))

        # mask   = generate_blindspot_mask(R1)

        # 内容保持为主， 局部平滑， 整体光照均匀

        # loss5 = 0.8 * similarity_loss(output1, R1) +  0.1 * local_smooth_loss(output1) +  0.1 * global_variance_loss(output1)  

        # loss6 = opt.loss_weights[0]*l_e(output1) + opt.loss_weights[1]*torch.mean(l_spa(X1, output1)) + opt.loss_weights[2]*l_tv(output1) + opt.loss_weights[3]*torch.mean(l_color(output1))

        # Ld = denoise_loss(I1, im1, mask)
        # Le = edge_loss(I1, im1)
        # Lu = unif_loss(I1)
        # loss5 = Ld + Le + Lu

        # 虚焦去模糊
        # loss6 = 

        # loss6 = clipiqa(I1) + brsique(I1)

        # loss_spa = torch.mean(spatial_consistency_loss(I3, im3))
        # loss_col = 5 * torch.mean(color_loss(I1))
        # loss_exp = 10 * torch.mean(exposure_loss(I3))
        # loss_tv = 200 * illumination_smoothing_loss(R3)
        # loss6 = loss_tv + loss_spa + loss_exp

        # loss6 = brsique(I1)


        # loss =  loss1 + loss2 + loss3 * 500 + loss4 + loss5 * 10
        # loss =  loss1 + loss2 + loss3 * 500 + loss4 + Ld * 10
        # loss =  loss1 + loss2 + loss3 * 500 + loss4
        #print(loss4)

        # 添加物理损失函数 L_phy = ||Î - I||_2^2 + μ||∇Î||_1 + ν(||∇σs||_1 + ||∇σc||_1)
        # 从模型的可微分层获取物理损失
        physical_loss = model.diff_layer.compute_physical_loss(I1, im1, mu=0.1, nu=0.1)
        
        # 获取模型输出的调制特征和注意力结果
        L1, el1, R1, X1, I1, modulated_features, Y = model(im1)
        
        # 提取CLIP特征用于频率损失计算
        _, resized_features = model.wavelet(im1)
        clip_embeddings = []
        for resized_feature in resized_features:
            embedding = model.clip_extractor(resized_feature)
            clip_embeddings.append(embedding)
        
        # 计算新的重建损失
        reconstruction_loss = model.compute_reconstruction_loss(R1, el1, I1)
        
        # 计算频率损失
        frequency_loss = model.compute_frequency_loss(modulated_features, clip_embeddings)
        
        # 计算一致性损失 (从Y中提取全局和局部路径)
        # 这里简化处理，假设Y的前半部分是全局路径，后半部分是局部路径
        batch_size, seq_len, dim = Y.shape
        global_path = Y[:, :seq_len//2, :]
        local_path = Y[:, seq_len//2:, :]
        consistency_loss = model.compute_consistency_loss(global_path, local_path)
        
        # 将所有损失添加到总损失中
        loss = loss1 + loss2 + loss3 * 500 + loss4 + physical_loss * 100 + reconstruction_loss * 50 + frequency_loss * 10 + consistency_loss * 5




        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_print = loss_print + loss.item()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                iteration, len(training_data_loader), loss_print, optimizer.param_groups[0]['lr']))
            loss_print = 0

def checkpoint(epoch):
    model_out_path = os.path.join(opt.save_folder, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
device = "cuda" if torch.cuda.is_available() else "cpu"

print('===> Loading datasets')
train_set = get_training_set(opt.data_train)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
test_set = get_eval_set(opt.data_val)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model ')
model= net().cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

writer = SummaryWriter(opt.logroot)
milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)

scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)
label_dir = opt.referance_val
score_best_epoch = 0
best_score = 1000
if not os.path.exists(opt.save_folder):
    os.mkdir(opt.save_folder)

clipiqa = pyiqa.create_metric("clipiqa", device="cuda")
brsique = pyiqa.create_metric('brisque', as_loss=False).cuda()

for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train()
    scheduler.step()
    if epoch % opt.snapshots == 0:
        checkpoint(epoch)

        avg_clipiqa = 0
        avg_brsique = 0
        n = 0

        for batch in testing_data_loader:
            n += 1
            with torch.no_grad():
                input, name = batch[0], batch[1]
            input = input.cuda()
            #print(name)

            # print(input.shape)

            with torch.no_grad():
                L, el, R, X ,I= model(input)
                # D = input - X
                I = torch.clamp(I, 0, 1)

                #EN_I = torch.clamp(EN_I, 0, 1)
                #print(EN_I.shape)

            im1 = I

            score_clipiqa = clipiqa(im1)
            score_brsique = brsique(im1)

            avg_clipiqa += score_clipiqa
            avg_brsique += score_brsique

        avg_clipiqa = avg_clipiqa / n
        avg_brsique = avg_brsique / n 

        avg_clipiqa = avg_clipiqa.item()
        avg_brsique = avg_brsique.item()
        
        if avg_clipiqa <= best_score:
            best_score = avg_clipiqa
            score_best_epoch = epoch
        
        writer.add_scalar('clipiqa', avg_clipiqa, epoch)
        writer.add_scalar('brsique', avg_brsique, epoch)

        print("===> Avg.clipiqa: {:.4f} ".format(avg_clipiqa))
        print("===> Avg.brsique: {:.4f} ".format(avg_brsique))

        print("===> best_score: {:.4f} ".format(best_score))
        print("===> score_best_epoch: {} ".format(score_best_epoch))


    """
    #if epoch == 0:
        checkpoint(epoch)
        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0
        n = 0
        loss_fn = lpips.LPIPS(net='alex')
        loss_fn.cuda()

        for batch in testing_data_loader:
            n += 1
            with torch.no_grad():
                input, name = batch[0], batch[1]
            input = input.cuda()
            #print(name)

            with torch.no_grad():
                L, el, R, X ,I= model(input)
                D = input- X
                I = torch.clamp(I, 0, 1)

                #EN_I = torch.clamp(EN_I, 0, 1)
                #print(EN_I.shape)

            #im2 = Image.open(label_dir + name[0].split('_')[0] + '.JPG').convert('RGB')
            im2 = Image.open(label_dir + name[0]).convert('RGB')
            (h, w) = im2.size
            im1 = I.squeeze(0)
            im1 = im1.permute(1,2,0).cpu() 

            im1 = np.array(im1)*255.
            im1 = im1.astype(np.uint8)
            im2 = np.array(im2)
            score_psnr = calculate_psnr(im1, im2)
            score_ssim = calculate_ssim(im1, im2)

            ex_p0 = I
            #ex_ref = lpips.im2tensor(lpips.load_image(label_dir + name[0].split('_')[0] + '.JPG'))
            ex_ref = lpips.im2tensor(lpips.load_image(label_dir + name[0]))
            ex_p0 = ex_p0.cuda()
            ex_ref = ex_ref.cuda()
            score_lpips = loss_fn.forward(ex_ref, ex_p0)
        
            avg_psnr += score_psnr
            avg_ssim += score_ssim
            avg_lpips += score_lpips
        
        avg_psnr = avg_psnr / n
        avg_ssim = avg_ssim / n
        avg_lpips = avg_lpips / n
        
        if avg_psnr >= best_score:
            best_score = avg_psnr
            score_best_epoch = epoch
        
        writer.add_scalar('psnr', avg_psnr, epoch)
        writer.add_scalar('ssim', avg_ssim, epoch)
        writer.add_scalar('lpips', avg_lpips, epoch)

        print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
        print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
        print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips.item()))

        print("===> best_score: {:.4f} ".format(best_score))
        print("===> score_best_epoch: {} ".format(score_best_epoch))

    """

source_file = os.path.join(opt.save_folder, f"epoch_{score_best_epoch}.pth")
target_file = os.path.join(opt.save_folder, "last_result.pth")

if os.path.exists(source_file):
    shutil.copy(source_file, target_file)


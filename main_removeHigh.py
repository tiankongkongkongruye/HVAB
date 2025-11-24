import os, pyiqa
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
# from net.Iformer_edit import net
# from net.Iformer_edit_attention_mamba import net
# from net.Iformer_edit_attention_Hist import net
from net.Iformer_Defect import net
from data import get_training_set, get_eval_set
from utils_Defect import *
from torch.utils.tensorboard import SummaryWriter


# python main_removeHigh.py --save_folder weights/Defect/ --logroot logs/Defect/ --data_train data/Defect/train/ --data_val data/Defect/test --referance_val data/Defect/test/ --lr 5e-6 --light_patch 64

# Training settings
parser = argparse.ArgumentParser(description='DE-Net')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=1e-5')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='100', help='learning rate decay type')
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

def specular_suppress_loss(L_map, threshold=0.9):
    mask = (L_map > threshold).float()
    penalty = ((L_map - threshold) * mask).pow(2)
    return penalty.mean()


def highlight_mask(L, threshold=0.9):
    return (L > threshold).float()

# 高光平滑损失
def smooth_loss_on_masked_region(R_clean, mask):
    # 计算梯度（注意尺寸会变）
    grad_x = torch.abs(R_clean[:, :, :, :-1] - R_clean[:, :, :, 1:])
    grad_y = torch.abs(R_clean[:, :, :-1, :] - R_clean[:, :, 1:, :])

    # 合并为单通道梯度图（取平均）
    grad = (grad_x.mean(dim=1, keepdim=True)[:, :, :-1, :] + grad_y.mean(dim=1, keepdim=True)[:, :, :, :-1]) / 2

    # 裁剪 mask 匹配梯度图尺寸
    mask_crop = mask[:, :, :-1, :-1]

    # 加权损失
    loss = (grad * mask_crop).mean()
    return loss

# 纹理保持损失
def texture_preserve_loss(R_clean, R_orig, mask):
    diff = (R_clean - R_orig).pow(2)
    return (diff * (1 - mask)).mean()  # 保持非高光区域尽量不变


def specular_suppress_loss(L, threshold=0.9):
    mask = (L > threshold).float()
    penalty = ((L - threshold) * mask).pow(2)
    return penalty.mean()

def reflectance_smoothness(R_clean, L):
    # Compute gradients
    grad_x = torch.abs(R_clean[:, :, :, :-1] - R_clean[:, :, :, 1:])
    grad_y = torch.abs(R_clean[:, :, :-1, :] - R_clean[:, :, 1:, :])

    grad = (grad_x.mean(dim=1, keepdim=True)[:, :, :-1, :] +
            grad_y.mean(dim=1, keepdim=True)[:, :, :, :-1]) / 2

    mask_crop = (L[:, :, :-1, :-1] > 0.9).float()
    loss = (grad * mask_crop).mean()
    return loss

def texture_preserve_loss(R_clean, R_orig, mask):
    diff = (R_clean - R_orig).pow(2)
    return (diff * (1 - mask)).mean()

def color_consistency_loss(R_clean, input):
    return F.l1_loss(R_clean, input)

def structure_preserve_loss(R_clean, input):
    grad_r = torch.abs(R_clean[:, :, :, :-1] - R_clean[:, :, :, 1:]) + \
             torch.abs(R_clean[:, :, :-1, :] - R_clean[:, :, 1:, :])
    grad_i = torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:]) + \
             torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :])
    return F.l1_loss(grad_r, grad_i)

def train():
    model.train()
    loss_print = 0

    l_e = L_exp(opt.light_patch,0.5)
    l_color = L_color()
    l_spa = L_spa()
    l_tv = L_TV()

    

    for iteration, batch in enumerate(training_data_loader, 1):

        input, file1 = batch[0], batch[1]
        input = input.cuda()
        # 生成两个降采样图像，构建多尺度训练样本
        im1, im2 = pair_downsampler(input)
        # 非线性增强，模拟不同光照条件
        im2 = gamma_correction(im2)
        # 随机拟合高光区域
        im4 = add_fake_highlight(im1)
        # im2 = highlight_correction(im2)
        #im2 = im1.pow(1 / opt.gama)
        #print(im1)
        im1 = im1.cuda()
        im2 = im2.cuda()
        im4 = im4.cuda()
        # 光照分量L，反射分量R，去噪中间结果X，增强后的光照图
        L1, el1, R1, R1_clean, X1, I1 = model(im1)
        L2, el2, R2, R2_clean, X2, _ = model(im2)
        L3, el3, R3, R3_clean, X3, I3 = model(input)
        L4, el4, R4, R4_clean, X4, I4 = model(im4)
        sub_r1, sub_r2 = pair_downsampler(I3)

        I1 = torch.clamp(I1, 0, 1)
        #el1 = torch.clamp(el1, 0, 1)
        X1 = torch.clamp(X1, 0, 1)
        R1 = torch.clamp(R1, 0, 1)
        R1_clean = torch.clamp(R1_clean, 0, 1)
        R2 = torch.clamp(R2, 0, 1)
        R2_clean = torch.clamp(R2_clean, 0, 1)
        #DI1 = torch.clamp(DI1, 0, 1)
        R4_clean = torch.clamp(R4_clean, 0, 1)
        
        # 反射一致性约束
        loss1 = C_loss(R1, R2) + C_loss(R1-R2, sub_r1-sub_r2) * 0.5
        #loss1 = C_loss(R1, R2)
        # 物理驱动重建损失
        loss2 = R_loss(L1, R1, im1, X1)
        # 感知损失
        loss3 = P_loss(im1, X1)
        # 复合正则：光照平滑，空间一致性，全变差约束，颜色保真
        loss4 = opt.loss_weights[0]*l_e(I1) + opt.loss_weights[1]*torch.mean(l_spa(X1, I1)) + opt.loss_weights[2]*l_tv(I1) + opt.loss_weights[3]*torch.mean(l_color(I1))

        # loss5 = highlight_loss(I3)

        # 高光mask
        mask1 = highlight_mask(L1)
        mask2 = highlight_mask(L4)

        # 高光区域内：希望R_clean < R（抑制）
        specular_loss = ((R1_clean - R1) * mask1).clamp(min=0).pow(2).mean() + ((R4_clean - R4) * mask2).clamp(min=0).pow(2).mean()

        # 非高光区域内：希望R_clean ≈ R（保持纹理）
        structure_loss = ((R1_clean - R1).pow(2) * (1 - mask1)).mean() + ((R4_clean - R4).pow(2) * (1 - mask2)).mean()

        # 合成高光区域，并使两者的ssim损失最小
        loss6 = F.l1_loss(R1_clean, R4_clean)

        loss5 = 2.0 * specular_loss + 1.0 * structure_loss + 1.0 * smooth_loss_on_masked_region(R1_clean, mask1) + 1.0 * texture_preserve_loss(R1_clean, R1, mask1) + loss6


        




        # loss5 = specular_suppress_loss(I1) + smooth_loss_on_masked_region(R1_clean, mask) + texture_preserve_loss(R1_clean, R1, mask)
        
    #     loss5 =  2.0 * specular_suppress_loss(L1) + \
    #    1.0 * reflectance_smoothness(R1_clean, L1) + \
    #    0.5 * texture_preserve_loss(R_clean, R, (L > 0.9).float()) + \
    #    1.0 * color_consistency_loss(R_clean, input) + \
    #    0.5 * structure_preserve_loss(R_clean, input)

        loss =  loss1 + loss2 + loss3 * 500 + loss4 + 100 * loss5
        #print(loss4)

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
best_score = 0
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

            with torch.no_grad():
                L, el, _, R, X ,I= model(input)
                D = input- X
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
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train()
    scheduler.step()
    if epoch % opt.snapshots == 0:
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
                L, el, R_org, R, X ,I= model(input)
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
        """

source_file = os.path.join(opt.save_folder, f"epoch_{score_best_epoch}.pth")
target_file = os.path.join(opt.save_folder, "last_result.pth")

if os.path.exists(source_file):
    shutil.copy(source_file, target_file)


import os, pyiqa
# import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# import torch
import glob
import cv2
import numpy as np
from PIL import Image
import torch

from data import get_eval_set
from torch.utils.data import DataLoader

# CLIP-IQA dependencies
# install with:
#   pip install clip-iqa
# from clip_iqa import ClipIQAMetric

# BRSIque dependencies
# install with:
#   pip install imquality
# from imquality import brisque

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

def metrics(im_dir):

    clipiqa = pyiqa.create_metric("clipiqa", device="cuda")
    brsique = pyiqa.create_metric('brisque', as_loss=False).cuda()

    avg_clipiqa = 0
    avg_brsique = 0
    n = 0

    test_set = get_eval_set(im_dir)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)


    for batch in testing_data_loader:
        n += 1
        with torch.no_grad():
            I, name = batch[0], batch[1]
            #input = gamma_correction(input)
        I = I.cuda()
       

        with torch.no_grad():
            I = torch.clamp(I, 0, 1)
            
        img = I

        assert img.dtype == torch.float32, "Input image must be float32"
        assert img.max() <= 1.0 and img.min() >= 0.0, "Input image must be in [0, 1]"

        score_clipiqa = clipiqa(img)
        score_brsique = brsique(img)

        print(name)
        print(score_clipiqa, score_brsique)

        avg_clipiqa += score_clipiqa
        avg_brsique += score_brsique


    avg_clipiqa = avg_clipiqa / n
    avg_brsique = avg_brsique / n      

    avg_clipiqa = avg_clipiqa.item()
    avg_brsique = avg_brsique.item()

    return avg_clipiqa, avg_brsique







"""

    for item in sorted(glob.glob(im_dir)):
        n += 1
        im1 = Image.open(item).convert('RGB') 

        name = item.split('/')[-1]

        # im1 = np.array(im1)
        # im1 = torch.clamp(im1, 0, 1)

        im1 = np.array(im1).astype(np.float32) / 255.0
        im1 = torch.from_numpy(im1).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]
        im1 = torch.clamp(im1, 0.0, 1.0)


        #print(im1.shape)
        # print(f"im1 的类型: {type(im1)}")  # 应为 numpy.ndarray 或 PIL.Image
        
        # 计算 CLIP-IQA 分数（越高越好）
        # score_clipiqa = clip_metric.score(im1)
        
        # score_brsique = brisque.score(im1)

        with torch.no_grad():
            score_clipiqa = clipiqa(im1)
            score_brsique = brsique(im1)

        

        print(name)
        print(score_clipiqa, score_brsique)
    
        avg_clipiqa += score_clipiqa
        avg_brsique += score_brsique

    avg_clipiqa = avg_clipiqa / n
    avg_brsique = avg_brsique / n 
    return avg_clipiqa, avg_brsique
    """

"""
def metrics(im_dir, label_dir):

    # clipiqa = pyiqa.create_metric("clipiqa", device="cuda")
    # brsique = pyiqa.create_metric('brisque', as_loss=False).cuda()

    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    n = 0
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()

    for item in sorted(glob.glob(im_dir)):
        n += 1
        im1 = Image.open(item).convert('RGB') 

        name = item.split('/')[-1]
        #name = name.split('_')[0] + '.png'
        #name = name.split('_')[0] + '.JPG'              # for SICE

        im2 = Image.open(label_dir + name).convert('RGB')
        (h, w) = im2.size
        im1 = im1.resize((h, w))  
        im1 = np.array(im1)
        im2 = np.array(im2)
        #print(im1.shape)
        score_psnr = calculate_psnr(im1, im2)
        score_ssim = calculate_ssim(im1, im2)

        ex_p0 = lpips.im2tensor(cv2.resize(lpips.load_image(item), (h, w)))
        ex_ref = lpips.im2tensor(lpips.load_image(label_dir + name))
        ex_p0 = ex_p0.cuda()
        ex_ref = ex_ref.cuda()
        #print(ex_p0.shape)
        score_lpips = loss_fn.forward(ex_ref, ex_p0)
        print(name)
        print(score_psnr,score_ssim,score_lpips)
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_lpips = avg_lpips / n
    return avg_psnr, avg_ssim, avg_lpips
"""

if __name__ == '__main__':

    """ im_dir = '/data2/lhq/results/sice/sci_difficult/*.png'
    label_dir = '/data2/lhq/dataset/pair_lie_dataset/PairLIE-testing-dataset/SICE-test/label/' """
    # im_dir = '/data2/lhq/PairLIE_edit/results/huawei96/I/*.jpg'
    # label_dir = '/data2/lhq/dataset/LSRW/Eval/Huawei/high/'

    # im_dir = 'results/SICE/I/*.JPG'
    # label_dir = '../dataset/LIE/SICE-test/label/'

    # im_dir = 'results/orginLOLv2/I/*.png'
    # label_dir = 'data/LOL-v2/Real_captured/Test/Normal/'

    # im_dir = 'results/orgin/I/*.png'
    # label_dir = 'data/LOL-v1/eval15/high/'

    # im_dir = 'data/middle/test/*.jpg'

    # im_dir = '/home/code/Zero-DCE/HTIW_enhanced_results/*.jpg'
    # im_dir = '/home/code/SCI/results/HTIW'

    # im_dir = "/home/code/LightenDiffusion/results/unpaired"


    im_dir = 'results/HTIW_CBAM_CSAM_RAG/I'
    # im_dir = 'data/HTIW/test'
    # im_dir = '/home/code/Zero-DCE/HTIW_enhanced_results'
    
    # im_dir = '/home/code/CLIP-LIT/inference_result'

    # im_dir = '/home/code/FSI/results/TOLED/output_images'

    # avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir)
    # print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    # print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
    # print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips.item()))

    

    avg_clipiqa, avg_brsique = metrics(im_dir)
    print("===> Avg.clipiqa: {:.4f} dB ".format(avg_clipiqa))
    print("===> Avg.brsique: {:.4f} ".format(avg_brsique))

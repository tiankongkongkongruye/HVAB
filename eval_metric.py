import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import argparse
#from thop import profile
#from net.net import net
# from net.lformer import net
# from net.Iformer_edit import net
# from net.Iformer_edit_attention import net
# from net.Iformer_edit_attention_Hist import net
# from net.Iformer_edit_retinex import net
# from net.Iformer_edit_attention_mamba import net
from net.Iformer_edit_channelattention import net
# from net.Iformer_edit_channelattention_CBAM import net
# from net.Iformer_edit_attention_HiLo import net
# from net.Iformer_edit_chanSp import net
# from net.Iformer_edit_crossAttention3 import net
# from net.Iformer_edit_CBAMattention import net
# from net.Iformer_edit_triAttention import net
from data import get_eval_set
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
from torchsummary import summary
import pyiqa

#python eval.py --data_test /data2/lhq/dataset/pair_lie_dataset/PairLIE-testing-dataset/MEF --output_folder /data2/lhq/PairLIE/results/MEF

parser = argparse.ArgumentParser(description='PairLIE')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
# parser.add_argument('--data_test', type=str, default='../dataset/LIE/LOL-test/raw')
# parser.add_argument('--data_test', type=str, default='../dataset/LIE/SICE-test/image')
parser.add_argument('--data_test', type=str, default='data/middle/test')
# parser.add_argument('--data_test', type=str, default='data/LOL-v1/eval15/low')
# parser.add_argument('--data_test', type=str, default='data/Defect')
# parser.add_argument('--model', default='weights/middle_test/epoch_4.pth', help='Pretrained base model')  
parser.add_argument('--model', default='weights/middle_quan/last_result.pth', help='Pretrained base model')  
parser.add_argument('--output_folder', type=str, default='results/middle_test')
opt = parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('===> Loading datasets')
test_set = get_eval_set(opt.data_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')
model = net().cuda()
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')

clipiqa = pyiqa.create_metric("clipiqa", device="cuda")
brsique = pyiqa.create_metric('brisque', as_loss=False).cuda()

def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print(count_parameters(model))
    print('\nEvaluation:')

    avg_clipiqa = 0
    avg_brsique = 0
    n = 0

    for batch in testing_data_loader:
        n += 1
        with torch.no_grad():
            input, name = batch[0], batch[1]
            #input = gamma_correction(input)
        input = input.cuda()
        print(name)

        with torch.no_grad():
            #print(input)
            L, _, R, X , I= model(input)
            D = input - X
            I = torch.clamp(I, 0, 1)
            R = torch.clamp(R, 0, 1)
            L = torch.clamp(L, 0, 1)       
            #I = torch.pow(L,1.2) * R  # default=0.2, LOL=0.14.
            # flops, params = profile(model, (input,))
            # print('flops: ', flops, 'params: ', params)


        img = I

        score_clipiqa = clipiqa(img)
        score_brsique = brsique(img)

        avg_clipiqa += score_clipiqa
        avg_brsique += score_brsique

        if not os.path.exists(opt.output_folder):
            os.mkdir(opt.output_folder)
            os.mkdir(opt.output_folder + '/L/')
            os.mkdir(opt.output_folder + '/R/')
            os.mkdir(opt.output_folder + '/I/')  
            os.mkdir(opt.output_folder + '/D/')                       

        L = L.cpu()
        R = R.cpu()
        I = I.cpu()
        D = D.cpu()        

        L_img = transforms.ToPILImage()(L.squeeze(0))
        R_img = transforms.ToPILImage()(R.squeeze(0))
        I_img = transforms.ToPILImage()(I.squeeze(0))                 
        D_img = transforms.ToPILImage()(D.squeeze(0))

        L_img.save(opt.output_folder + '/L/' + name[0])
        R_img.save(opt.output_folder + '/R/' + name[0])
        I_img.save(opt.output_folder + '/I/' + name[0])  
        D_img.save(opt.output_folder + '/D/' + name[0])   

    avg_clipiqa = avg_clipiqa / n
    avg_brsique = avg_brsique / n      

    avg_clipiqa = avg_clipiqa.item()
    avg_brsique = avg_brsique.item()        

    print("===> Avg.clipiqa: {:.4f} dB ".format(avg_clipiqa))
    print("===> Avg.brsique: {:.4f} ".format(avg_brsique))      

    torch.set_grad_enabled(True)

eval()



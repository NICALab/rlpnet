from torch.autograd import Variable
import torch
import argparse
import os
from torchvision.utils import save_image
import skimage.io as skio
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from glob import glob

from models import GeneratorRRDB, GeneratorLSTMRRDB
from utils import iter_inference, PSNR, SSIM

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, default=1, help="Number of image channels")
parser.add_argument("--output_path", type=str, default="../", help="directory of output path")
parser.add_argument("--single", type=bool, default=False, help="high res. image width")
parser.add_argument("--step_size", type=int, default=1, help="step size(pixels) of plane-to-plane")
parser.add_argument("--switch", type=str, default='lstm', help="None, lstm")
parser.add_argument("--test_data", type=str, default='real_beads', help="old_celegans_bin4, real_beads")
parser.add_argument("--test_patch", type=str, default='whole', help="whole, head, body1, body2, tail")
parser.add_argument("--saved_model_path", type=str, default="../", help="saved model path")
parser.add_argument("--test_data_path", type=str, default="../", help="test data path")
parser.add_argument("--test_name", type=str, default='', help="name of test file")
parser.add_argument("--test_list", type=list, default=[9], help="test AB/BA list")
opt = parser.parse_args()
print(opt)

# Window position, C.elegans
old_celegans_bin4_body1 = (56, 87)
old_celegans_bin4_body2 = (157, 87)
old_celegans_bin4_head = (259, 45)
old_celegans_bin4_tail = (8, 38)

transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
]

def denormalize(tensors):
    mean = 0.5
    std = 0.5
    tensors.mul_(std).add_(mean)
    return torch.clamp(tensors, 0, 255)


os.makedirs(opt.output_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
if opt.switch == 'lstm':
    generator = torch.nn.DataParallel(GeneratorLSTMRRDB(channels=opt.channels, filters=32, num_res_blocks=8)).to(device)
else:
    generator = torch.nn.DataParallel(GeneratorRRDB(channels=opt.channels, filters=32, num_res_blocks=8)).to(device)

for epoch in range(2000, 2050, 10):
    temp_list = glob(opt.saved_model_path + '/G_{}_*.pth'.format(epoch))
    while len(temp_list) == 0:
        epoch -= 1
        temp_list = glob(opt.saved_model_path + '/G_{}_*.pth'.format(epoch))
    epoch_list = sorted(temp_list)
    generator.load_state_dict(torch.load(epoch_list[-1]))

    generator.eval()
    transform = transforms.Compose(transforms_)
    psnr = PSNR()
    ssim = SSIM()

    with torch.no_grad():
        volume = skio.imread(opt.test_data_path)
        input_shape = (volume.shape[1], volume.shape[2])

        if opt.test_data == 'old_celegans_bin4':
            win_size_y, win_size_x = 128, 128
            if opt.test_patch == 'body1':
                start_y, start_x = old_celegans_bin4_body1
            elif opt.test_patch == 'body2':
                start_y, start_x = old_celegans_bin4_body2
            elif opt.test_patch == 'tail':
                start_y, start_x = old_celegans_bin4_tail
            elif opt.test_patch == 'head':
                start_y, start_x = old_celegans_bin4_head
        elif opt.test_data == 'real_beads':
            win_size_y, win_size_x = input_shape 
            start_y, start_x = 0, 0   
        
        for infer_dir in ['AB', 'BA']:
            initial_plane_list = opt.test_list

            for initial_plane in initial_plane_list:
                if infer_dir == 'AB':
                    infer_len = 6
                else:
                    infer_len = 6

                for z in range(0,infer_len):
                    if z == 1:
                        continue

                    target = []
                    lent = 2
                    if z == infer_len-1:
                        lent = 1
                    
                    if infer_dir == 'AB':
                        for t in range(lent):
                            target_ = volume[initial_plane + opt.step_size*t + opt.step_size*2*z, start_y:start_y+win_size_y, start_x:start_x+win_size_x]
                            target_ = transform(Image.fromarray(np.uint8(target_)))
                            target_ = Variable(target_).to(device).unsqueeze(0)
                            target.append(target_)
                    else:
                        for t in range(lent):
                            target_ = volume[initial_plane + opt.step_size - opt.step_size*t - opt.step_size*2*z, start_y:start_y+win_size_y, start_x:start_x+win_size_x]
                            target_ = transform(Image.fromarray(np.uint8(target_)))
                            target_ = Variable(target_).to(device).unsqueeze(0)
                            target.append(target_)
                    
                    if z==0:
                        output_img = target
                        target_cat = torch.cat((target[0], target[1]), 0)
                        output_cat = torch.cat((target[0], target[1]), 0)
                    else:
                        output = iter_inference(generator, output_img[0], output_img[1], num_iter=2*z-1, switch=opt.switch)
                        output_cat = torch.cat((output_cat, output), 0)
                        target_cat = torch.cat((target_cat, target[0]), 0)

                if opt.single:
                    factor_img = 1
                else:
                    factor_img = 4

                output_cat = denormalize(output_cat) * 1
                target_cat = denormalize(target_cat) * 1
                output_cat_gray = output_cat
                target_cat_gray = target_cat
                diff_cat_gray = (output_cat - target_cat) * factor_img
                zero = torch.zeros_like(diff_cat_gray)
                one = torch.ones_like(diff_cat_gray)

                diff_cat = torch.cat([torch.where(diff_cat_gray < 0, zero, diff_cat_gray), zero, torch.where(diff_cat_gray > 0, zero, -1 * diff_cat_gray)], dim=1)
                output_cat = torch.cat([torch.zeros_like(output_cat), output_cat, torch.zeros_like(output_cat)], dim=1)
                target_cat = torch.cat([torch.zeros_like(target_cat), target_cat, torch.zeros_like(target_cat)], dim=1)

                if opt.single == True:
                    for i in range(len(output_cat)):
                        if i > 1:
                            print('<infer_dir {} / initial_plane {} / current step {}>'.format(infer_dir, initial_plane, i))
                            print('PSNR between input & GT : %.2f' % psnr(output_cat_gray[0, :, :, :] * 255.0, target_cat_gray[i, :, :, :] * 255.0).item())
                            print('PSNR between output & GT : %.2f' % psnr(output_cat_gray[i ,:, :, :] * 255.0, target_cat_gray[i, :, :, :] * 255.0).item())
                            print('SSIM between input & GT : %.4f' % ssim(output_cat_gray[0, :, :, :].permute(1,2,0) * 255.0, target_cat_gray[i, :, :, :].permute(1,2,0) * 255.0))
                            print('SSIM between output & GT : %.4f' % ssim(output_cat_gray[i, :, :, :].permute(1,2,0) * 255.0, target_cat_gray[i, :, :, :].permute(1,2,0) * 255.0))
                            
                else:
                    img = torch.cat((output_cat_gray[::,:,:,:], target_cat_gray[::,:,:,:], torch.abs(diff_cat_gray[::,:,:,:])), -2).cpu()
                    save_image(img, "{}/HL_{}_z{}_epoch{}_{}_{}_{}.png".format(opt.output_path, infer_dir, initial_plane, epoch, opt.test_patch, opt.test_data, opt.test_name))


            
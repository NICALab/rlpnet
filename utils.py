import os
import sys
import math
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
import skimage.io as skio
from torch.autograd import Variable
from torchvision.utils import save_image
from skimage.metrics import normalized_root_mse
import scipy.optimize as opt

########################################################################
#-----------------------Functions for Inference------------------------#
########################################################################

def denormalize(tensors):
    mean = 0.5
    std = 0.5
    tensors.mul_(std).add_(mean)
    return torch.clamp(tensors, 0, 255)


def infer_trainset(root, root2, batches_done, ind_type, ind_list, crop_pos, G, move_dir, step_size, switch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        ]
    transform = transforms.Compose(transforms_)

    img_bead_list = glob(root + '/*.tif')
    img_tot_list = [img_bead_list]
    
    selected_list = img_tot_list[ind_type]
    volume = skio.imread(selected_list[ind_list])
    z = crop_pos[0]
    nz = 0

    if move_dir == 'AB':
        while nz <= 3 and (z + step_size + step_size*2*nz) < volume.shape[0]:
            target = []
            for t in range(2):
                target_ = volume[z + step_size*t + step_size*2*nz, crop_pos[1]:crop_pos[1]+32, crop_pos[2]:crop_pos[2]+32]
                target_ = transform(Image.fromarray(np.uint8(target_)))
                target_ = Variable(target_).to(device).unsqueeze(0)
                target.append(target_)

            if nz==0:
                output_img = target
                target_cat = torch.cat((target[0], target[1]), 0)
                output_cat = torch.cat((target[0], target[1]), 0)
            else:
                output = iter_inference(G, output_img[0], output_img[1], num_iter=2*nz-1, switch=switch)
                output_cat = torch.cat((output_cat, output), 0)
                target_cat = torch.cat((target_cat, target[0]), 0)
            nz += 1
    else:
        while nz <= 3 and (z - step_size - step_size*2*nz) >= 0:
            target = []
            for t in range(2):
                target_ = volume[z - step_size*t - step_size*2*nz, crop_pos[1]:crop_pos[1]+32, crop_pos[2]:crop_pos[2]+32]
                target_ = transform(Image.fromarray(np.uint8(target_)))
                target_ = Variable(target_).to(device).unsqueeze(0)
                target.append(target_)

            if nz==0:
                output_img = target
                target_cat = torch.cat((target[0], target[1]), 0)
                output_cat = torch.cat((target[0], target[1]), 0)
            else:
                output = iter_inference(G, output_img[0], output_img[1], num_iter=2*nz-1, switch=switch)
                output_cat = torch.cat((output_cat, output), 0)
                target_cat = torch.cat((target_cat, target[0]), 0)
            nz += 1
    
    output_cat = denormalize(output_cat) * 1
    target_cat = denormalize(target_cat) * 1
    diff_cat = (output_cat - target_cat) * 4
    zero = torch.zeros_like(diff_cat)

    diff_cat = torch.cat([torch.where(diff_cat < 0, zero, diff_cat), zero, torch.where(diff_cat > 0, zero, -1 * diff_cat)], dim=1)
    output_cat = torch.cat([torch.zeros_like(output_cat), output_cat, torch.zeros_like(output_cat)], dim=1)
    target_cat = torch.cat([torch.zeros_like(target_cat), target_cat, torch.zeros_like(target_cat)], dim=1)

    img = torch.cat((output_cat, target_cat, diff_cat), -2).cpu()
    save_image(img, root2 + "/images/training/batchesDone_%d_inputPlane_%d_%s.png" % (batches_done, z, move_dir))


def draw_train_curve(root2, batches_done, loss_d, loss_g):
    x = [i for i in range(1, len(loss_d) + 1)]
    plt.clf()
    plt.plot(x, loss_d, lw=0.75, color='blue', label='loss_D')
    plt.plot(x, loss_g, lw=0.75, color='red', label='loss_G')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='upper left', prop={'size': 6})
    plt.savefig(root2 + '/images/loss_curve/lossDG_batchesDone_%d.png' % batches_done)

def draw_lossG_curve(root2, batches_done, raw_list, name):
    x = [i for i in range(1, len(raw_list) + 1)]
    plt.clf()
    plt.plot(x, raw_list, lw=0.75, color='blue', label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='upper left', prop={'size': 6})
    plt.savefig(root2 + '/images/loss_curve/%s_batchesDone_%d.png' % (name, batches_done))


def iter_inference(model, input1, input2, num_iter, switch):
    cur_output1 = input1
    cur_output2 = input2
    for i in range(num_iter):
        if switch=='lstm':
            if i==0:
                cur_output, hidden_state = model(torch.cat((cur_output1, cur_output2), dim=1))
            else:
                cur_output, hidden_state = model(torch.cat((cur_output1, cur_output2), dim=1), hidden_state)
        else:
            cur_output = model(torch.cat((cur_output1, cur_output2), dim=1))
        cur_output1 = cur_output2
        cur_output2 = cur_output
    
    return cur_output


########################################################################
#----------------------------Eval Metrics------------------------------#
########################################################################

class PSNR:
    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

def _ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.cpu().numpy().astype(np.float64)
    img2 = img2.cpu().numpy().astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()

class SSIM:
    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return _ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(_ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return _ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")
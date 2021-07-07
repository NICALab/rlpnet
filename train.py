import matplotlib
matplotlib.use('Agg')
import argparse
import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
import numpy as np
import datetime
import time
import os
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP

from models import weights_init_normal, GeneratorRRDB, GeneratorLSTMRRDB, Discriminator
from datasets import denormalize, ImageDataset
from utils import iter_inference, draw_train_curve, draw_lossG_curve, infer_trainset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default='../', help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=128, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=128, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="batch interval between model checkpoints")
parser.add_argument("--warmup_epochs", type=int, default=100, help="number of batches with pixel-wise loss only")
parser.add_argument("--step_size", type=int, default=1, help="step size(pixels) of plane-to-plane")
parser.add_argument("--multi_step", type=int, default=7, help="number of multi-step(1~7) for training")
parser.add_argument("--step_up", type=str, default='0-150-200-250-300-350-400', help="step-up point for multi-step(single step='0') for training")
parser.add_argument("--root2", type=str, default='../', help="directory of training results")
parser.add_argument("--batches_done", type=int, default=None, help="bacthes done, to resume training")
parser.add_argument("--multi_step_init", type=int, default=0, help="initial step shifted, to resume training")
parser.add_argument("--switch", type=str, default='lstm', help="None or lstm")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.root2 + "/images/training", exist_ok=True)
os.makedirs(opt.root2 + "/images/loss_curve", exist_ok=True)
os.makedirs(opt.root2 + "/saved_models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
if opt.switch == 'lstm':
    G = torch.nn.DataParallel(GeneratorLSTMRRDB(channels=opt.channels, filters=32, num_res_blocks=8)).cuda()
else:
    G = torch.nn.DataParallel(GeneratorRRDB(channels=opt.channels, filters=32, num_res_blocks=8)).cuda()
D = Discriminator(input_shape=(opt.channels, *input_shape)).to(device)

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
pixelwise_loss = torch.nn.L1Loss().to(device)
cycle_loss = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    G.load_state_dict(torch.load(opt.root2 + "/saved_models/G_%d_%d.pth" % (opt.epoch, opt.batches_done)))
    D.load_state_dict(torch.load(opt.root2 + "/saved_models/D_%d_%d.pth" % (opt.epoch, opt.batches_done)))
else:
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)


# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr / 4, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
device = torch.device('cuda')

dataset = ImageDataset("%s" % opt.dataset_name, input_shape=input_shape, step_size=opt.step_size, multi_step=opt.multi_step)
dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

loss_d_list = []
loss_g_list = []

loss_GAN_list = []
loss_cycle_list = []
loss_pixelwise_list = []

prev_time = time.time()

step_up = list(map(lambda x:int(x), opt.step_up.split('-')))

if len(step_up) != opt.multi_step:
    raise ValueError

multi_step = opt.multi_step_init
for epoch in range(opt.epoch, opt.n_epochs):
    multi_step = multi_step + 1 if epoch in step_up else multi_step
    for i, imgs in enumerate(dataloader):
        cur_step = random.randint(1, multi_step) # cur_step is random value from 1 to multi_step

        batches_done = epoch * len(dataloader) + i

        # Configure model input
        real_A1  = Variable(imgs["input_"][0].type(Tensor))
        real_A2 = Variable(imgs["input_"][1].type(Tensor))
        real_B1  = Variable(imgs["target_"][cur_step].type(Tensor))
        real_B2  = Variable(imgs["target_"][cur_step - 1].type(Tensor))

        move_dir = str(imgs["move_dir_"][0])
        ind_type = int(imgs["indexOfType_"][0])
        ind_list = int(imgs["indexOfList_"][0])
        crop_pos = [int(imgs["cropPosition_"][i][0]) for i in range(3)]

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A1.size(0), *D.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A1.size(0), *D.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G.train()

        optimizer_G.zero_grad()
        
        # GAN loss
        fake_B1 = iter_inference(G, real_A1, real_A2, cur_step, switch=opt.switch)
        loss_GAN = criterion_GAN(D(fake_B1), valid)

        # Pixelwise loss
        loss_pixelwise = pixelwise_loss(fake_B1, real_B1)
        loss_pixelwise_list.append(loss_pixelwise.item())

        if epoch < opt.warmup_epochs:
            # Warm-up (pixel-wise loss only)
            loss_pixelwise.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f, cur_step: %d]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_pixelwise.item(), cur_step)
            )

            if batches_done % opt.sample_interval == 0:
                # inference of training set for warmup epochs
                infer_trainset(opt.dataset_name, opt.root2, batches_done, ind_type, ind_list, crop_pos, G, move_dir, opt.step_size, switch=opt.switch)
            
            if opt.checkpoint_interval != -1 and batches_done % opt.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(G.state_dict(), opt.root2 + "/saved_models/G_%d_%d.pth" % (epoch, batches_done))
                torch.save(D.state_dict(), opt.root2 + "/saved_models/D_%d_%d.pth" % (epoch, batches_done))
            
            continue

        # Cycle loss
        back_step = random.randint(1, cur_step)
        if back_step == cur_step:
            real_mid  = real_A1
        else:
            real_mid  = Variable(imgs["target_"][cur_step - 1 - back_step].type(Tensor))
        loss_cycle = cycle_loss(iter_inference(G, fake_B1, real_B2, back_step, switch=opt.switch), real_mid)
        loss_cycle_list.append(loss_cycle.item())
        loss_GAN_list.append(loss_GAN)

        # Total loss
        loss_G = 0.001 * loss_GAN + 0.01 * loss_cycle + loss_pixelwise

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator
        # -----------------------

        optimizer_D.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D(real_B1), valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = criterion_GAN(D(fake_B1.detach()), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        print(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f, cycle: %f, cur_step : %d, multi_step : %d] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_pixelwise.item(),
                loss_cycle.item(),
                cur_step,
                multi_step,
                time_left,
            )
        )

        loss_d_list.append(loss_D.item())
        loss_g_list.append(loss_G.item())

        if batches_done % opt.sample_interval == 0:
            # Save image grid with inputs and outputs
            infer_trainset(opt.dataset_name, opt.root2, batches_done, ind_type, ind_list, crop_pos, G, move_dir, opt.step_size, switch=opt.switch)


        if opt.checkpoint_interval != -1 and batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G.state_dict(), opt.root2 + "/saved_models/G_%d_%d.pth" % (epoch, batches_done))
            torch.save(D.state_dict(), opt.root2 + "/saved_models/D_%d_%d.pth" % (epoch, batches_done))

    if epoch % 10 == 0:
        # plot loss curves
        draw_train_curve(opt.root2, batches_done, loss_d_list, loss_g_list)
        draw_lossG_curve(opt.root2, batches_done, loss_pixelwise_list, 'loss_pix')
        draw_lossG_curve(opt.root2, batches_done, loss_GAN_list, 'loss_GAN')
        draw_lossG_curve(opt.root2, batches_done, loss_cycle_list, 'loss_cycle')

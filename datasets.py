import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import skimage.io as skio
from glob import glob
import torchvision.transforms.functional as TF


def denormalize(tensors):
    mean = 0.5
    std = 0.5
    tensors.mul_(std).add_(mean)
    return torch.clamp(tensors, 0, 255)

class RandomRotationx90:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

rotation_transform = RandomRotationx90()
transforms_ = [
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    rotation_transform,

    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
]

class ImageDataset(Dataset):
    def __init__(self, root, input_shape, step_size, multi_step):
        self.img_beads_list = glob(root + '/*.tif')
        self.img_tot_list = [self.img_beads_list]
        
        self.step_size = step_size
        self.multi_step = multi_step
        self.input_shape = input_shape
        self.transform = transforms.Compose(transforms_)

    def __getitem__(self, index):
        ind = np.random.randint(len(self.img_tot_list))
        selected_list = self.img_tot_list[ind]
        
        volume = skio.imread(selected_list[index%len(selected_list)])
        volume_shape = volume.shape
        
        rand_init = [random.randint(0, volume_shape[0]-self.step_size-1), 0, 0]
        bool_move_neg = (rand_init[0] >= (self.multi_step * self.step_size))
        bool_move_pos = (rand_init[0] <= (volume_shape[0] - self.multi_step * self.step_size - 1 - self.step_size))
        
        ind_tgt = np.random.randint(2)
        if not bool_move_pos:
            ind_tgt = 1
        if not bool_move_neg:
            ind_tgt = 0

        input_ = []
        target_ = []
        if ind_tgt == 0:
            for n_input in range(2):
                input_.append(volume[rand_init[0] + n_input * self.step_size,\
                        rand_init[1]:rand_init[1]+self.input_shape[0], rand_init[2]:rand_init[2]+self.input_shape[1]])
            for n_step in range(0, self.multi_step + 1):
                target_.append(volume[rand_init[0] + self.step_size + n_step * self.step_size, \
                                rand_init[1]:rand_init[1] + self.input_shape[0],
                                rand_init[2]:rand_init[2] + self.input_shape[1]])
        else:
            for n_input in range(2):
                input_.append(volume[rand_init[0] + self.step_size - n_input * self.step_size,\
                        rand_init[1]:rand_init[1]+self.input_shape[0], rand_init[2]:rand_init[2]+self.input_shape[1]])
            for n_step in range(0, self.multi_step + 1):
                target_.append(volume[rand_init[0] - n_step * self.step_size, \
                                rand_init[1]:rand_init[1] + self.input_shape[0],
                                rand_init[2]:rand_init[2] + self.input_shape[1]])

        input_list = []
        target_list = []
        seed = np.random.randint(2147483647)
        for num in range(len(input_)):
            random.seed(seed)
            torch.manual_seed(seed)
            input_list.append(self.transform(Image.fromarray(np.uint8(input_[num]))))
        for num in range(len(target_)):
            random.seed(seed)
            torch.manual_seed(seed)
            target_list.append(self.transform(Image.fromarray(np.uint8(target_[num]))))

        move_dir_ = 'AB' if ind_tgt == 0 else 'BA'

        return {"input_": input_list, "target_": target_list, "move_dir_": move_dir_, "indexOfType_": ind, "indexOfList_": index%len(selected_list), "cropPosition_": rand_init}

    def __len__(self):
        return len(self.img_tot_list[0]) * 1
        # last value is approximate value, considering each volume size!


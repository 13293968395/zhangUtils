__version__ = '0.1.1'

import torchvision
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from skimage import io
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch.nn.functional as F
import os

def show(img):

    if torch.is_tensor(img) | isinstance(img, np.ndarray):
        img = transforms.ToPILImage()(img)
    
    img.show()

def shape(tensor):
    print(tensor.size())

def saveImage(img, img_name='test', img_path='./images/'):
    #img 图片
    #img_name 仅需图片的名字，无需'.jpg'
    #img_path 图片存储路径
    if os.path.exists(img_path) == False:
        os.makedirs(img_path)

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    
    if torch.is_tensor(img):
        torchvision.utils.save_image(img, img_path + img_name + ".jpg") 
        return

    img.save(img_path)

def readPic(pic_path, typeNumber=1):

    #typyNumber返回的照片参数
    #                       1:ndarry 2:tensor 3:PIL.Image
    img = io.imread(pic_path)

    if type == 1:
        return img

    if type == 2:
        return transforms.ToTensor()(img)
    
    # PIL Image
    if type == 3:
        return Image.open(pic_path).convert('RGB')
    

def ssim(cleanImagePath, dehazeImagePath):
    clean_image = transforms.ToTensor()(io.imread(cleanImagePath))
    dehaze_image = transforms.ToTensor()(io.imread(dehazeImagePath))
    ssim_val = structural_similarity(clean_image.permute(1, 2, 0).cpu().numpy(), 
                                     dehaze_image.permute(1, 2, 0).numpy(), data_range=1, multichannel=True)
    return ssim_val

def psnr(cleanImagePath, dehazeImagePath):
    clean_image = transforms.ToTensor()(io.imread(cleanImagePath))
    dehaze_image = transforms.ToTensor()(io.imread(dehazeImagePath))
    psnr_val = 10 * torch.log10(1 / F.mse_loss(dehaze_image, clean_image))
    return psnr_val.item()
    



  
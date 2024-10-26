__version__ = '0.1.0'

"""
实现你自己的轮子的功能
"""

import torchvision
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

def show(img):

    if torch.is_tensor(img) | isinstance(img, np.ndarray):
        img = transforms.ToPILImage()(img)
    
    img.show()

def shape(tensor):
    print(tensor.size())

def save_image(img, img_path='./images/'):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    
    if torch.is_tensor(img):
        torchvision.utils.save_image(img, img_path)
        return

    img.save(img_path)
    



  
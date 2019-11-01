from torch import nn
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import random
import glob
import io
from PIL import Image

import numpy as np
import PIL.Image as pil_image
import glob
import cv2
import numpy as np
import PIL
# from multiprocessing import Pool
from torch.utils.data import Dataset

import tensorflow as tf
patch_size, stride = 128, 20
aug_times = 2

batch_size = 5
'''
class Aug(object):
    def __init__(self, img_path):
        super(Aug, self).__init__()
        self.image_files = sorted(glob.glob(img_path + '/*'))

    def __itemget__(self, idx):
        clean_image = pil_image.open(self.image_files[idx]).convert('RGB')

 '''   
    
def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    
def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 1)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    patches = []
    '''
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        
    '''   
        # extract patches
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = img[i:i+patch_size, j:j+patch_size,:]
            for k in range(0, aug_times):
                x.reshape(patch_size,patch_size,c)
                x_aug = data_aug(x, mode=np.random.randint(0, 8))
                patches.append(x_aug)
    return patches

def datagenerator(data_dir='BSDS500/images/train', verbose=True):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.jpg')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:    
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    #data = np.array(data, dtype='uint8')
    #data = np.expand_dims(data, axis=3)
    #discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    #data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data

data1 = datagenerator(data_dir='BSDS500/images/train')
print(data1[0].shape)
print(len(data1))
'''
img = PIL.Image.fromarray(data1[4700])
plt.imshow(img)
'''
i=1
for data in data1:
    img = PIL.Image.fromarray(data)
    img.save(os.path.join('final_input', '{}.png'.format(i)))
    i=i+1


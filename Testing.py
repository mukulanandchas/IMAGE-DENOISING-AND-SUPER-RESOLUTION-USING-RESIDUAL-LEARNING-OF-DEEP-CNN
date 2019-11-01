import argparse
import os
import io
import numpy as np
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from model import RES
import matplotlib.pyplot as plt
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from PIL import ImageFilter as IF
import skimage.measure
from skimage.measure import block_reduce

if not os.path.exists('256'):
    os.makedirs('256')

model = RES()
state_dict = model.state_dict()
#print(state_dict)
for n, p in torch.load('newoutput/nearest-test-6_epoch_10.pth', map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)

model = model.to(device)
model.eval()

filename = os.path.basename('256/pool.png').split('.')[0]
descriptions = ''
#os.path.exists('BSDS300/images/test/3096.jpg')
input = pil_image.open('256/pool.png').convert('RGB')

plt.imshow(input)
print(input.height)

input = input.resize((256, 256))

mdownsampling_factor = 2
mgaussian_noise_level = 0

original_width = input.width
original_height = input.height
#input = input.filter(IF.GaussianBlur(1))
#descriptions += 'gblur_'
#input.save(os.path.join('output', '{}{}.png'.format(filename, descriptions)))


#input = input.resize((input.width // mdownsampling_factor,
#                              input.height // mdownsampling_factor))
input = np.array(input).astype(np.float32)
input=skimage.measure.block_reduce(input, (2,2,1), np.mean)

input=pil_image.fromarray(input.clip(0.0, 255.0).astype(np.uint8))    
    
descriptions += 'scaled{}'.format(mdownsampling_factor)
input.save(os.path.join('256', '{}{}.png'.format(filename, descriptions)))


noise = np.random.normal(0.0, mgaussian_noise_level, (input.height, input.width, 3)).astype(np.float32)
input = np.array(input).astype(np.float32) + noise
descriptions += '_noise_l{}'.format(mgaussian_noise_level)
pil_image.fromarray(input.clip(0.0, 255.0).astype(np.uint8)).save(os.path.join('256', '{}{}.png'.format(filename, descriptions)))
input/=255.0
plt.imshow(input)

input = pil_image.open(os.path.join('256', '{}{}.png'.format(filename, descriptions))).convert('RGB')
input = input.resize((original_width, original_height), resample=pil_image.NEAREST)
descriptions += '_sr_s{}'.format(mdownsampling_factor)
input.save(os.path.join('256', '{}{}.png'.format(filename, descriptions)))
input = np.array(input).astype(np.float32)
input /= 255.0
#input = pil_image.open(os.path.join('output', '{}{}.png'.format(filename, descriptions))).convert('RGB')

'''
#if mgaussian_noise_level is not None:
noise = np.random.normal(0.0, mgaussian_noise_level, (input.height, input.width, 3)).astype(np.float32)
input = np.array(input).astype(np.float32) + noise
descriptions += '_noise_l{}'.format(mgaussian_noise_level)
pil_image.fromarray(input.clip(0.0, 255.0).astype(np.uint8)).save(os.path.join('output', '{}{}.png'.format(filename, descriptions)))
input /= 255.0
'''
input = transforms.ToTensor()(input).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(input)
    #pred = pred-input
#|||
output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
output = pil_image.fromarray(output, mode='RGB')
output.save(os.path.join('256', '{}{}_{}.png'.format(filename, descriptions, 'test-6')))



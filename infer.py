# !pip install torchsummary
# !pip install torchgeometry
# !pip install segmentation-models-pytorch

import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['ALBUMENTATIONS_SKIP_VERSION_CHECK'] = '1'

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = smp.Unet(
    encoder_name="efficientnet-b3",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=3     
)

val_transformation = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

color_dict= {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}

def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output) 

checkpoint = torch.load('unet-cel.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model.to(device)

import sys

def parse_kwargs(argv):
    kwargs = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=")
            kwargs[key.lstrip("--")] = value
    return kwargs

argv = sys.argv[1:]
kwargs = parse_kwargs(argv)
img_path = kwargs.get("image_path")

print(img_path)
ori_img = cv2.imread(img_path)
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
ori_w = ori_img.shape[0]
ori_h = ori_img.shape[1]
img = cv2.resize(ori_img, (256, 256))
transformed = val_transformation(image=img)
input_img = transformed["image"]
input_img = input_img.unsqueeze(0).to(device)
with torch.no_grad():
    output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
mask = cv2.resize(output_mask, (ori_h, ori_w))
mask = np.argmax(mask, axis=2)
mask_rgb = mask_to_rgb(mask, color_dict)
mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

alpha = 0.5  
im = cv2.addWeighted(ori_img, 1 - alpha, mask_rgb, alpha, 0)

cv2.imwrite("prediction/{}".format("result.jpeg"), im)

# for i in os.listdir("test"):
#     img_path = os.path.join("test", i)
#     ori_img = cv2.imread(img_path)
#     ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
#     ori_w = ori_img.shape[0]
#     ori_h = ori_img.shape[1]
#     img = cv2.resize(ori_img, (256, 256))
#     transformed = val_transformation(image=img)
#     input_img = transformed["image"]
#     input_img = input_img.unsqueeze(0).to(device)
#     with torch.no_grad():
#         output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
#     mask = cv2.resize(output_mask, (ori_h, ori_w))
#     mask = np.argmax(mask, axis=2)
#     mask_rgb = mask_to_rgb(mask, color_dict)
#     mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

#     alpha = 0.5  
#     im = cv2.addWeighted(ori_img, 1 - alpha, mask_rgb, alpha, 0)

#     cv2.imwrite("prediction/{}".format(i), im)


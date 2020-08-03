from quick_start.coarseAlignFeatMatch import CoarseAlign
import sys

sys.path.append('utils/')
import utils.outil as outil

sys.path.append('model/')
import model.model as model

import PIL.Image as Image
import os
import numpy as np
import torch
from torchvision import transforms
import warnings
import torch.nn.functional as F
import kornia.geometry as tgm

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import matplotlib.pyplot as plt


## composite image
def get_Avg_Image(Is, It):
    Is_arr, It_arr = np.array(Is), np.array(It)
    Imean = Is_arr * 0.5 + It_arr * 0.5
    return Image.fromarray(Imean.astype(np.uint8))


# %%

resumePth = 'model/pretrained/MegaDepth_Theta1_Eta001_Grad1_0.774.pth'  ## model for visualization
kernelSize = 7

Transform = outil.Homography
nbPoint = 4

## Loading model
# Define Networks
network = {'netFeatCoarse': model.FeatureExtractor(),
           'netCorr': model.CorrNeigh(kernelSize),
           'netFlowCoarse': model.NetFlowCoarse(kernelSize),
           'netMatch': model.NetMatchability(kernelSize),
           }

for key in list(network.keys()):
    network[key].cuda()
    typeData = torch.cuda.FloatTensor

# loading Network 
param = torch.load(resumePth)
msg = 'Loading pretrained model from {}'.format(resumePth)
print(msg)

for key in list(param.keys()):
    network[key].load_state_dict(param[key])
    network[key].eval()

# %% md

### Without alignment

# %%
def align_image(source_path, output_filename, destination_path, base_path):
    I1 = Image.open(source_path).convert('RGB')
    I2 = Image.open(destination_path).convert('RGB')

    ### 7 scales, setting ransac parameters

    nbScale = 7
    coarseIter = 10000
    coarsetolerance = 0.05
    minSize = 400
    imageNet = True  # we can also use MOCO feature here
    scaleR = 1.2

    coarseModel = CoarseAlign(nbScale, coarseIter, coarsetolerance, 'Homography', minSize, 1, True, imageNet, scaleR)

    coarseModel.setSource(I1)
    coarseModel.setTarget(I2)

    I2w, I2h = coarseModel.It.size
    featt = F.normalize(network['netFeatCoarse'](coarseModel.ItTensor))

    #### -- grid
    gridY = torch.linspace(-1, 1, steps=I2h).view(1, -1, 1, 1).expand(1, I2h, I2w, 1)
    gridX = torch.linspace(-1, 1, steps=I2w).view(1, 1, -1, 1).expand(1, I2h, I2w, 1)
    grid = torch.cat((gridX, gridY), dim=3).cuda()
    warper = tgm.HomographyWarper(I2h, I2w)

    bestPara, InlierMask = coarseModel.getCoarse(np.zeros((I2h, I2w)))
    bestPara = torch.from_numpy(bestPara).unsqueeze(0).cuda()

    ### Coarse Alignment

    flowCoarse = warper.warp_grid(bestPara)
    I1_coarse = F.grid_sample(coarseModel.IsTensor, flowCoarse)
    I1_coarse_pil = transforms.ToPILImage()(I1_coarse.cpu().squeeze())
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('Source Image (Coarse)')
    plt.imshow(I1_coarse_pil)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('Target Image')
    plt.imshow(I2)
    plt.subplot(1, 3, 3)
    plt.title('Overlapped Image')
    plt.imshow(I2)
    plt.imshow(get_Avg_Image(I1_coarse_pil, coarseModel.It))
    plt.show()

    ### Fine Alignment

    featsSample = F.normalize(network['netFeatCoarse'](I1_coarse.cuda()))

    corr12 = network['netCorr'](featt, featsSample)
    flowDown8 = network['netFlowCoarse'](corr12, False)  ## output is with dimension B, 2, W, H

    flowUp = F.interpolate(flowDown8, size=(grid.size()[1], grid.size()[2]), mode='bilinear')
    flowUp = flowUp.permute(0, 2, 3, 1)

    flowUp = flowUp + grid

    flow12 = F.grid_sample(flowCoarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()

    I1_fine = F.grid_sample(coarseModel.IsTensor, flow12)
    I1_fine_pil = transforms.ToPILImage()(I1_fine.cpu().squeeze())
    if not os.path.exists(os.path.join(base_path, "output")):
        os.mkdir(os.path.join(base_path, "output"))
    I1_fine_pil.save(os.path.join(base_path, "output", output_filename))
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('Source Image (Fine Alignment)')
    plt.imshow(I1_fine_pil)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('Target Image')
    plt.imshow(I2)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title('Overlapped Image')
    plt.imshow(get_Avg_Image(I1_fine_pil, coarseModel.It))
    plt.show()

paths = [
"../colored_photos_and_originals/low_lighting_colored_aligned_galaxy",
    # "../colored_photos_and_originals/high_lighting_colored_aligned_galaxy",
    #      "../colored_photos_and_originals/high_lighting_colored_aligned_s10",
    #      "../colored_photos_and_originals/low_lighting_colored_aligned_s10"
]

# for path in paths:
#     for index, f in enumerate(sorted(os.listdir(path))):
#         align_image(os.path.join(path, f), '../colored_photos_and_originals/originals_color - Copy/' + f)

align_image('out.jpg', 'out_aligned.jpg','0.jpg', '.')

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np
import h5py
import argparse
import random
import torch.utils.model_zoo as model_zoo

#datasets
from Datasets.ILSVRC import ImageNetDataset_val

#models
import Methods.AGCAM.ViT_for_AGCAM as ViT_Ours
import timm
from Methods.LRP.ViT_LRP import vit_base_patch16_224 as LRP_vit_base_patch16_224

#methods
from Methods.AGCAM.AGCAM import AGCAM
from Methods.LRP.ViT_explanation_generator import LRP
from Methods.AttentionRollout.AttentionRollout import VITAttentionRollout

parser = argparse.ArgumentParser(description='save heatmaps in h5')
parser.add_argument('--method', type=str, choices=['agcam', 'lrp', 'rollout'])
parser.add_argument('--save_root', type=str, required=True)
parser.add_argument('--data_root', type=str, required=True)
args = parser.parse_args()



MODEL = 'vit_base_patch16_224'
DEVICE = 'cuda'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set seed
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(777)
if device == DEVICE:
    gc.collect()
    torch.cuda.empty_cache()
print("device: " +  device)

# root to save the h5 file
save_root = args.save_root
save_name=""

# root of the dataset
data_root = args.data_root

# transformation for ILSVRC
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])
unnormalize = transforms.Compose([
    transforms.Normalize([0., 0., 0.], [1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize([-0.5, -0.5, -0.5], [1., 1., 1.,])
])

validset = ImageNetDataset_val(
    root_dir=data_root,
    transforms=test_transform,
)

# create model and the method
state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=False, map_location=device)
class_num=1000
save_name +="ILSVRC"

if args.method=="agcam":
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = AGCAM(model)
    save_name +="_agcam"
elif args.method=="lrp":
    model = LRP_vit_base_patch16_224(device=device, num_classes=class_num).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = LRP(model, device=device)
    save_name+="_lrp"
elif args.method=="rollout":
    model = timm.create_model(MODEL, pretrained=True, num_classes=class_num).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = VITAttentionRollout(model, device=device)
    save_name+='_rollout'


print("save the data in ", save_root)

# make h5py file
file = h5py.File(os.path.join(save_root, save_name+".hdf5"), 'w')
file.create_group('label')  # gruop to save the class label of each image
file.create_group('image')  # group to save the original image
file.create_group('cam')    # group to save the heatmap visualization

g_label=file['label']
g_image=file['image']
g_cam = file['cam']


validloader = DataLoader(
    dataset = validset,
    batch_size=1,
    shuffle = False,
)

with torch.enable_grad():
    for data in tqdm(validloader):
        image = data['image'].to(device)
        label = data['label'].to(device)
        filename = data['filename']

        # generate heatmap
        prediction, heatmap = method.generate(image, label)

        # resize the heatmap
        resize = transforms.Resize((224, 224))
        heatmap = resize(heatmap[0])

        # normalize the heatmap from 0 to 1
        heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())
        heatmap = heatmap.detach().cpu().numpy()

        #unnormalize the image to save the original input image
        image = unnormalize(image)
        image = image.detach().cpu().numpy()

        # save the class label, input image and the heatmap
        g_image.create_dataset(filename[0], data=image)
        g_label.create_dataset(filename[0], data=label.detach().cpu().numpy())
        g_cam.create_dataset(filename[0], data=heatmap)
file.close()